import tenseal as ts
import time
import psutil

# Step 1: Create a TenSEAL context
# Generate a CKKS context with default parameters
def key_generation_task():
    print(f"1): {psutil.virtual_memory()[3]}")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    print(f"2): {psutil.virtual_memory()[3]}")
    # Set global scale for encoding
    context.global_scale = 2**40
    print(f"3): {psutil.virtual_memory()[3]}")
    # Generate the public and private keys
    context.generate_galois_keys()
    print(f"4): {psutil.virtual_memory()[3]}")
    context.generate_relin_keys()
    print(f"5): {psutil.virtual_memory()[3]}")
    return context


timeGenKey = time.time()

before = psutil.virtual_memory()[3]
context = key_generation_task()
after = psutil.virtual_memory()[3]
print(before, after)
print((after-before)/(1024*1024))


timeGenKey = time.time() - timeGenKey

# Step 2: Encrypt data
# Plain data
data = [1000000.5, 200000000.5, 300000000.5]
print(f"Original data: {data}")

timeEncryptData = time.time()
# Encrypt the data
encrypted_vector = ts.ckks_vector(context, data)

timeEncryptData = time.time() - timeEncryptData

timeCalCEncryptData = time.time()
# Step 3: Perform computations on encrypted data
# Perform addition
encrypted_vector += [1.0, 1.0, 1.0]

timeCalCEncryptData = time.time() -  timeCalCEncryptData

timeDecryptData = time.time()
# Step 4: Decrypt the result
# Decrypt the computed data
decrypted_result = encrypted_vector.decrypt()
timeDecryptData = time.time() - timeDecryptData
print(f"Decrypted result: {decrypted_result}")

print(f"Key Gen: {timeGenKey}")
print(f"Encryption: {timeEncryptData}")
print(f"Calc on encrypted data: {timeCalCEncryptData}")
print(f"Decryption: {timeDecryptData}")
