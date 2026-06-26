import requests
import base64
from PIL import Image
from io import BytesIO

# Port 8080 is Spring Boot backend
SB_URL = "http://localhost:8080"
headers = {
    "Content-Type": "application/json"
}

# 1. Create a mock transparent image (RGBA)
print("Creating a mock RGBA image (transparent background with an opaque red square inside)...")
w, h = 100, 100
mock_img = Image.new("RGBA", (w, h), (0, 0, 0, 0)) # fully transparent
# Put a red square in the middle
for y in range(30, 70):
    for x in range(30, 70):
        mock_img.putpixel((x, y), (255, 0, 0, 255)) # opaque red

buffered_img = BytesIO()
mock_img.save(buffered_img, format="PNG")
img_b64 = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

# 2. Create a brush mask over the red square (e.g. to remove it)
print("Creating a mask to erase a part of the red square...")
mask_img = Image.new("L", (w, h), 0)
for y in range(40, 50):
    for x in range(40, 50):
        mask_img.putpixel((x, y), 255)

buffered_mask = BytesIO()
mask_img.save(buffered_mask, format="PNG")
mask_b64 = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")

# 3. Test Spring Boot Authentication (Spring Boot requires JWT token, let's login/register a test user)
print("\n--- Step 1: Registering Test User on Spring Boot ---")
auth_url = SB_URL + "/api/auth/register"
payload = {
    "username": "testuser_fixes",
    "email": "testuser_fixes@example.com",
    "password": "Password123!",
    "phone": "1234567890"
}
try:
    resp = requests.post(auth_url, json=payload)
    print("Registration status:", resp.status_code)
    data = resp.json()
    token = data.get("data", {}).get("token")
except Exception as e:
    # If already exists, try logging in
    print("Registration failed/user exists, trying Login...")
    login_url = SB_URL + "/api/auth/login"
    payload = {
        "email": "testuser_fixes@example.com",
        "password": "Password123!"
    }
    resp = requests.post(login_url, json=payload)
    print("Login status:", resp.status_code)
    data = resp.json()
    token = data.get("data", {}).get("token")

if not token:
    print("Error: Could not obtain JWT token from Spring Boot!")
    exit(1)

print("Obtained JWT Token successfully.")
headers["Authorization"] = f"Bearer {token}"

# 4. Test Background Removal
print("\n--- Step 2: Testing Background Removal via Spring Boot ---")
url = SB_URL + "/api/remove-bg"
payload = {
    "image": img_b64
}
resp = requests.post(url, json=payload, headers=headers)
print("Response status:", resp.status_code)
if resp.status_code == 200:
    res_data = resp.json()
    out_b64 = res_data.get("image")
    out_img = Image.open(BytesIO(base64.b64decode(out_b64)))
    print(f"Background Removal SUCCESS! Mode: {out_img.mode}, Size: {out_img.size}")
else:
    print("Failed background removal:", resp.text)

# 5. Test Object Removal (Inpaint) with Alpha channel preservation
print("\n--- Step 3: Testing Object Removal (SimpleLama) via Spring Boot ---")
url = SB_URL + "/api/inpaint"
payload = {
    "image": img_b64,
    "mask": mask_b64,
    "prompt": "remove object"
}
resp = requests.post(url, json=payload, headers=headers)
print("Response status:", resp.status_code)
if resp.status_code == 200:
    res_data = resp.json()
    out_b64 = res_data.get("image")
    out_img = Image.open(BytesIO(base64.b64decode(out_b64)))
    print(f"Object Removal SUCCESS! Mode: {out_img.mode}, Size: {out_img.size}")
    # Check if transparency was preserved!
    alpha_vals = [out_img.getpixel((x, y))[3] for x in range(10) for y in range(10)]
    print(f"Transparency preservation test: Alpha values at border (first 10 pixels): {alpha_vals}")
    if all(a == 0 for a in alpha_vals):
        print("PASS: Transparent background was successfully preserved!")
    else:
        print("FAIL: Transparency was lost or altered.")
else:
    print("Failed object removal:", resp.text)

print("\nValidation complete.")
