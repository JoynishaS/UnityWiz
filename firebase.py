import pyrebase

config = {
    "apiKey": "AIzaSyAKavNUziDSEElPWKAVlaPXZrKoNdKDocs",
    "authDomain": "unitywiz.firebaseapp.com",
    "databaseURL": "https://unitywiz-default-rtdb.firebaseio.com",
    "projectId": "unitywiz",
    "storageBucket": "unitywiz.appspot.com",
    "messagingSenderId": "420253959046",
    "appId": "1:420253959046:web:d75c640674d00974f7a207",
    "measurementId": "G-3VFPD7P61H"
}

firebase = pyrebase.initialize_app(config)

cloudpath = "/Test.pdf"

firebase.storage().child(cloudpath).download(filename="UnityManual.pdf")