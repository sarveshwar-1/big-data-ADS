import requests
import os
import time

# Configuration
EXPRESS_UPLOAD_URL = "http://localhost:3000/upload"
WEBHDFS_BASE_URL = "http://localhost:9870/webhdfs/v1"
TEST_FILENAME = "test_upload_file.txt"
TEST_CONTENT = b"Hello, this is a test file uploaded via Express to HDFS!"

def create_test_file():
    with open(TEST_FILENAME, "wb") as f:
        f.write(TEST_CONTENT)
    print(f"Created local test file: {TEST_FILENAME}")

def upload_file():
    print(f"Uploading {TEST_FILENAME} to {EXPRESS_UPLOAD_URL}...")
    with open(TEST_FILENAME, "rb") as f:
        files = {'file': (TEST_FILENAME, f)}
        try:
            response = requests.post(EXPRESS_UPLOAD_URL, files=files)
            if response.status_code == 200:
                print("Upload successful!")
                return True
            else:
                print(f"Upload failed with status code: {response.status_code}")
                print(response.text)
                return False
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the Express server. Is it running on port 3000?")
            return False

def check_hdfs_file():
    # Check if file exists in HDFS
    hdfs_url = f"{WEBHDFS_BASE_URL}/{TEST_FILENAME}?op=GETFILESTATUS"
    print(f"Checking HDFS status at: {hdfs_url}")
    
    try:
        response = requests.get(hdfs_url)
        if response.status_code == 200:
            status = response.json().get("FileStatus", {})
            print("File found in HDFS!")
            print(f"Owner: {status.get('owner')}")
            print(f"Size: {status.get('length')} bytes")
            print(f"Permission: {status.get('permission')}")
            
            if status.get('length') == len(TEST_CONTENT):
                print("SUCCESS: File size matches!")
                return True
            else:
                print(f"FAILURE: File size mismatch. Expected {len(TEST_CONTENT)}, got {status.get('length')}")
                return False
        elif response.status_code == 404:
            print("File NOT found in HDFS.")
            return False
        else:
            print(f"Error checking HDFS: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to NameNode. Is it running on port 9870?")
        return False

def cleanup():
    if os.path.exists(TEST_FILENAME):
        os.remove(TEST_FILENAME)
        print(f"Removed local test file: {TEST_FILENAME}")

def main():
    try:
        create_test_file()
        if upload_file():
            # Give it a moment just in case, though the upload should be synchronous
            time.sleep(1)
            if check_hdfs_file():
                print("\n--- TEST PASSED ---")
            else:
                print("\n--- TEST FAILED: File not found or incorrect in HDFS ---")
        else:
            print("\n--- TEST FAILED: Upload failed ---")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
