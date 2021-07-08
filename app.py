#########################################################################################
### handle imports
import time
#import json
import sys

from flask import Response
from flask import Flask, request
#from flask_cors import CORS
from flask import jsonify

#########################################################################################


#########################################################################################
### init flask
app = Flask(__name__)
#CORS(app)

#########################################################################################


#########################################################################################
### API routes

# route for steering the vehicle
@app.route('/steer', methods=['POST'])
def post_image():


    message = "ping"


    return jsonify(message)

# route for generated video ouput
@app.route("/")
def get_image():


    message = "ping"
    print("ping")


    return jsonify(message)


#########################################################################################


#########################################################################################
### start the flask app
if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port="6475", ssl_context='adhoc', debug=True,
            threaded=True, use_reloader=False)
        #socketio.run(app, host='0.0.0.0', port=6475, debug=True)
    except KeyboardInterrupt:
        #check for exit
        time.sleep(1)
        print("exit program")
        sys.exit(0)
