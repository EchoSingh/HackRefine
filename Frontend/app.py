from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('./Frontend/try.html')

if __name__ == '__main__':
    app.run()
