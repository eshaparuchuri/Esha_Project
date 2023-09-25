import cv2
import tensorflow as tf
import numpy as np
import pygame, os, random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
model = tf.keras.models.load_model(r'webapp\models\MobileNet.h5', compile=False)

# Set up video capture device
cap = cv2.VideoCapture(0)

# Define labels
labels = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 'fighting', 'hugging',
          'laughing', 'listening_to_music', 'running', 'sitting', 'sleeping', 'texting', 'using_laptop']


def livestreaming():
    global i;
    i=0
    print(i)
    while True:
        # Read frame from video capture device
        ret, frame = cap.read()

        # Preprocess the image
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Make prediction
        prediction = model.predict(input_frame)[0]

        # Get predicted label
        predicted_label = labels[np.argmax(prediction)]
        i=i+1
        print(i)
        if i==35:
            if predicted_label=="sleeping" or predicted_label=="eating" or predicted_label=="calling" or predicted_label=="fighting":
                path = r"webapp\siren\alert"
                file = os.path.join(path, random.choice(os.listdir(path)))
                pygame.mixer.init()
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                # msg = 'Dear Sir/Madam,'
                # otp = "Your students identified with an activity is :"
                # t='Regards,'
                # t1='Activity Recognition Services.'
                # mail_content = msg + '\n' + otp+ predicted_label +'.'+'\n'+'\n'+t+'\n'+t1
                # sender_address = 'eshap0808@gmail.com' # e mail open cheyandi browser lo
                # sender_pass = ''
                # receiver_address = "eshaparuchuri26@gmail.com"
                # message = MIMEMultipart()
                # message['From'] = sender_address
                # message['To'] = receiver_address
                # message['Subject'] = 'Human Activity Services'
                
                # message.attach(MIMEText(mail_content, 'plain'))
                # session = smtplib.SMTP('smtp.gmail.com', 587)
                # session.starttls()
                # session.login(sender_address, sender_pass)
                # text = message.as_string()
                # session.sendmail(sender_address, receiver_address, text)
                # session.quit()
                i=0;
            # else:
            #     pass
        # Overlay label onto image
        cv2.putText(frame, predicted_label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show video frame
        cv2.imshow('frame', frame)

        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture device and close window
    cap.release()
    cv2.destroyAllWindows()
