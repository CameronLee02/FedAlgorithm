import threading
import time
import tenseal as ts
import numpy as np
import torch
from models.test import test_fun

class ServerNodeClass():
    def __init__(self, node_id, network):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.node_list = None
        self.route = None
    
    #This function collects all the nodes that are in the network
    def getNodeList(self, node_list):
        self.node_list = node_list 
    
    def receiveMessage(self, sender_id, message):
        if len(message.keys()) != 1:
            return
        
        if "VALIDATED_NODE_ROUTE" in message.keys() and sender_id in self.node_list.keys():
            self.route = message["VALIDATED_NODE_ROUTE"]
    
    #This function is used to simulate the central server sending a list of the participating to all the nodes 
    def sendOutListOfNodes(self):
        print("Sending out list of nodes")
        start = time.time()
        self.network.messageAllNodesExcludeServer(0, {"NODE_LIST" : list(self.node_list.keys())})
        print(f"Time taken : {time.time()-start}")
    
    # CKKS Context Setup
    def create_ckks_context(self):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context
    
    def decryptWeights(self, encrypted_weights, context, original_shapes, text_widget, client_count, overhead_info):
        start_time = time.time()
        self.network.updateText("Decrypting aggregated weights using CKKS decryption...", text_widget)
        decrypted_weights = {}
        for name, enc_weight_chunks in encrypted_weights.items():
            decrypted_flat = []
            for enc_weight in enc_weight_chunks:
                decrypted_flat.extend(enc_weight.decrypt())

            decrypted_array = (np.array(decrypted_flat, dtype=np.float32) / client_count).reshape(original_shapes[name])
            decrypted_weights[name] = torch.tensor(decrypted_array, dtype=torch.float32)

            self.network.logWeightStats({name: decrypted_weights[name]}, "Decrypted Weights", text_widget)

            if self.network.checkForNan({name: decrypted_weights[name]}, "Decrypted Weights", text_widget):
                raise ValueError(f"NaN detected in decrypted weights: {name}")

        decryption_time = time.time() - start_time
        self.network.updateText(f"Decryption completed in {decryption_time:.4f} seconds.", text_widget)
        overhead_info["decryption_times"].append(decryption_time)
        return decrypted_weights

    def threaded_client_training(self, node_object, node_id, dataset_train, dict_party_user, net_glob, context, text_widget, received_encrypted_weights, results, G, visualisation_canvas, visualisation_ax, colours, pos, args, overhead_info):
        encrypted_weights, local_weights, loss = node_object.client_training(
            node_id,
            dataset_train,
            dict_party_user,
            net_glob,
            text_widget,
            context,
            args,
            overhead_info,
            G, #this parameter and below are used for route visualisation except received_encrypted_weights
            visualisation_canvas,
            visualisation_ax,
            colours,
            pos,
            received_encrypted_weights
        )
        results[node_id] = (encrypted_weights, local_weights, loss)
    
    def trainingProcess(self, net_glob, arg, dataset_train, dict_party_user, text_widget, visualisation_canvas, visualisation_ax, overhead_info, ax1, ax2, canvas):
        context = self.create_ckks_context()

        net_glob.train()

        epoch_losses = []
        epoch_accuracies = []

        start_total_time = time.time()
        
        for iter in range(arg.epochs):
            epoch_start_time = time.time()
            self.network.updateText(f'+++ Epoch {iter + 1} starts +++', text_widget)

            self.sendOutListOfNodes()
            print(self.route)
            #print(f"Number of threads running rn: {threading.active_count()}")

            colours, pos, G = self.network.displayNetwork(self.route, visualisation_canvas, visualisation_ax)
            
            local_losses = []
            original_shapes = {name: weight.shape for name, weight in net_glob.state_dict().items()}

            received_encrypted_weights = None
            results = {}
            threads = []

            for idx, node_id in enumerate(self.route):
                node_object = self.node_list[node_id]
                thread = threading.Thread(
                    target=self.threaded_client_training,
                    args=(node_object, node_id, dataset_train, dict_party_user, net_glob, context, text_widget, received_encrypted_weights, results, G, visualisation_canvas, visualisation_ax, colours, pos, arg, overhead_info)
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            client_count = 0

            for idx, user in enumerate(self.route):
                encrypted_weights, local_weights, loss = results[user]
                client_count += 1

                if received_encrypted_weights is not None:
                    received_encrypted_weights = self.network.aggregateEncryptedWeights(
                        received_encrypted_weights,
                        encrypted_weights,
                        client_count,
                        text_widget
                    )
                else:
                    received_encrypted_weights = encrypted_weights
                colours[idx] = "green"
                self.network.updateDisplayNetwork(G, visualisation_canvas, visualisation_ax, colours, pos)

                local_losses.append(loss)

            self.network.updateText('Final client sending aggregated encrypted weights to server.', text_widget)
            decrypted_weights = self.decryptWeights(received_encrypted_weights, context, original_shapes, text_widget, client_count, overhead_info)

            if self.network.checkForNan(decrypted_weights, "Global Model Weights", text_widget):
                raise ValueError("NaN detected in global model weights before updating.")
            
            self.network.logWeightStats(decrypted_weights, "Global Model Weights", text_widget)

            net_glob.load_state_dict(decrypted_weights)
            self.network.updateText('Server has updated the global model with final aggregated weights.', text_widget)

            net_glob.eval()
            acc_train, _ = test_fun(net_glob, dataset_train, arg)
            epoch_losses.append(np.mean(local_losses))
            epoch_accuracies.append(acc_train)

            self.network.updateText(f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}', text_widget)
            self.network.updatePlots(epoch_losses, epoch_accuracies, ax1, ax2, canvas)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            overhead_info["epoch_times"].append(epoch_duration)
            visualisation_ax.clear() #used to clear/reset the network visualisation window
            visualisation_canvas.draw()

        overhead_info["total_time"] = time.time() - start_total_time
        return acc_train