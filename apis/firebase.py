# import os
import pyrebase as pb
# from apis.config import cfg

firebaseConfig = {
    "apiKey": "AIzaSyCo65hFywwv7ZH66SHOjZsT6Co3tQa5JOU",
    "authDomain": "vses-9a738.firebaseapp.com",
    "databaseURL": "https://vses-9a738-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "projectId": "vses-9a738",
    "storageBucket": "vses-9a738.appspot.com",
    "messagingSenderId": "466523576690",
    "appId": "1:466523576690:web:7ec7ce219d89f40233a5eb"
}

firebase = pb.initialize_app(firebaseConfig)
storage = firebase.storage()
