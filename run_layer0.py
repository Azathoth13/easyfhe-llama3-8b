#!/usr/bin/env python3
"""Stable project entrypoint for the encrypted feature-major Layer0."""

from llama3fhe.application.runner import main

if __name__ == "__main__":
    main(layer0_preset=True)
