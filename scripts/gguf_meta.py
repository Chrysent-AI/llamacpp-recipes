"""Read only the GGUF v3 metadata/tensor directory; never map weight data.

Layout verified against the pinned llama.cpp gguf-py/gguf/gguf_reader.py.
This deliberately supports only the little-endian recipe artifact.
"""
import struct


def inspect(path):
    with open(path, "rb") as f:
        def number(fmt):
            size = struct.calcsize(fmt)
            raw = f.read(size)
            if len(raw) != size:
                raise ValueError("Truncated GGUF directory")
            return struct.unpack("<" + fmt, raw)[0]

        def string(keep=True):
            size = number("Q")
            if size > 16 * 1024 * 1024:
                raise ValueError("Unreasonable GGUF string length")
            if not keep:
                f.seek(size, 1)
                return None
            raw = f.read(size)
            if len(raw) != size:
                raise ValueError("Truncated GGUF string")
            return raw.decode("utf-8")

        def value(kind, keep=True):
            formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                       6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
            if kind in formats:
                return number(formats[kind])
            if kind == 8:
                return string(keep)
            if kind == 9:
                subtype, count = number("I"), number("Q")
                if count > 2000000 or subtype == 9:
                    raise ValueError("Unsupported GGUF array")
                for _ in range(count):
                    value(subtype, False)
                return None
            raise ValueError("Unsupported GGUF metadata type")

        if f.read(4) != b"GGUF" or number("I") != 3:
            raise ValueError("Expected little-endian GGUF v3")
        tensors, entries = number("Q"), number("Q")
        if tensors > 100000 or entries > 100000:
            raise ValueError("Unreasonable GGUF directory size")
        metadata = {}
        for _ in range(entries):
            key = string()
            keep = key in ("general.architecture", "general.name") or key.endswith(
                (".block_count", ".nextn_predict_layers", ".context_length"))
            item = value(number("I"), keep)
            if keep:
                metadata[key] = item
        names = set()
        for _ in range(tensors):
            names.add(string())
            dims = number("I")
            if dims > 4:
                raise ValueError("Unsupported GGUF tensor rank")
            for _ in range(dims):
                number("Q")
            number("I")
            number("Q")
        return metadata, names


def require_mtp(metadata, names):
    arch = metadata.get("general.architecture")
    layers = metadata.get(f"{arch}.block_count", 0)
    # The pinned Qwen35 loader uses the last block as the single nextn block.
    if arch != "qwen35" or metadata.get(f"{arch}.nextn_predict_layers") != 1:
        raise ValueError("MTP requires qwen35 with one nextn block; inspect the exact GGUF, no fallback")
    prefix = f"blk.{layers - 1}."
    required = [prefix + suffix + ".weight" for suffix in (
        "nextn.eh_proj", "nextn.enorm", "nextn.hnorm", "attn_norm",
        "post_attention_norm", "attn_q", "attn_k", "attn_v", "attn_output",
        "attn_q_norm", "attn_k_norm", "ffn_gate", "ffn_down", "ffn_up")]
    missing = set(required) - names
    if missing:
        raise ValueError("MTP tensors missing: " + ", ".join(sorted(missing)))
    return sorted(name for name in names if ".nextn." in name)
