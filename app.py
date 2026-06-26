from flask import Flask, send_from_directory

app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    print("Starting Studio AI Python Backend on http://localhost:8000")
    app.run(port=8000, debug=True)
