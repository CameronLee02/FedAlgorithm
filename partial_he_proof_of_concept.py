import random

class SimpleHE:
    def __init__(self, modulus=1000000007):
        # Using a large prime modulus for better properties under modulo
        self.modulus = modulus
        self.key = random.randint(1, modulus - 1)  # Ensuring the key is within the valid range

    def encrypt(self, value):
        # Encrypt by adding the key and taking modulo
        return (value + self.key) % self.modulus

    def decrypt(self, value):
        # Decrypt by subtracting the key and then modulo
        # Adding modulus before taking modulo to ensure non-negative result
        return (value - self.key + self.modulus) % self.modulus

    def add(self, enc_value1, enc_value2):
        # Perform addition on encrypted values and ensure it wraps around using the same modulus
        return (enc_value1 + enc_value2) % self.modulus

# Usage
he = SimpleHE()
value1 = 10
value2 = 20

enc_value1 = he.encrypt(value1)
enc_value2 = he.encrypt(value2)

# Encrypted addition
enc_sum = he.add(enc_value1, enc_value2)
decrypted_sum = he.decrypt(enc_sum)

print("Encrypted Value 1:", enc_value1)
print("Encrypted Value 2:", enc_value2)
print("Encrypted Sum:", enc_sum)
print("Decrypted Sum:", decrypted_sum)  # Should output 30
