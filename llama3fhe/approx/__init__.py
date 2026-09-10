"""Plaintext mathematics: what the encrypted kernels approximate.

Nothing here touches EasyFHE, a crypto context or a ciphertext. These modules
define the functions the FHE kernels implement (so tests can compare against
them), fit or load their polynomial coefficients, and compute the derived
quantities the schedule needs (degrees, multiplicative depth).

The split matters because the two halves have different notions of being
correct: a module here is correct when it matches the mathematics, while a
kernel under ``operators/`` is additionally correct only when its limb
budgets, rescales and ciphertext lifetimes are right.
"""
