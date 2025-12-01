import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import requests
import json

BASE = "http://127.0.0.1:8000/ai"

def test_register():
    files = {
        "leaf_image": open("sample_images/leafs/leaf1.jpeg", "rb"),
        "bark_image": open("sample_images/bark/bark1.jpg", "rb"),
        "tree_image": open("sample_images/tree/mangotree.jpeg", "rb"),
    }

    data = {
        "owner_id": "user123",
        "gps_lat": 19.12345,
        "gps_lng": 72.98765,
    }

    res = requests.post(f"{BASE}/register-tree", data=data, files=files)
    print("REGISTER RESPONSE:")
    print(res.json())

    return res.json().get("tree_id")


def test_verify_leaf(tree_id):
    files = {
        "leaf_image": open("sample_images/leaf/leaf1.jpg", "rb"),
    }

    data = {
        "user_id": "user123",
        "gps_lat": 19.12345,
        "gps_lng": 72.98765,
    }

    res = requests.post(f"{BASE}/verify-tree", data=data, files=files)
    print("VERIFY (LEAF) RESPONSE:")
    print(res.json())


def test_verify_bark(tree_id):
    files = {
        "bark_image": open("sample_images/bark/bark1.jpg", "rb"),
    }

    data = {
        "user_id": "user123",
        "gps_lat": 19.12345,
        "gps_lng": 72.98765,
    }

    res = requests.post(f"{BASE}/verify-tree", data=data, files=files)
    print("VERIFY (BARK) RESPONSE:")
    print(res.json())


if __name__ == "__main__":
    print("=== Testing Register → Verify ===")
    tid = test_register()
    print("Tree ID:", tid)
    test_verify_leaf(tid)
    test_verify_bark(tid)
