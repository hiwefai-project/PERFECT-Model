import numpy as np
import smtplib
from email.mime.text import MIMEText
import matplotlib.pyplot as plt

def flood_alert(flow_acc, threshold, email_recipient):
    """
    Identifica zone a rischio di inondazione e invia un'allerta.
    
    args: 
        flow_acc: array di accumulo di flusso
        threshold: soglia per zone a rischio
        email_recipient: email destinatario
    """
    
    risk_zones = flow_acc > threshold
    alert_count = np.sum(risk_zones)
    if alert_count > 0:
        print(f"Allerta: {alert_count} zone a rischio di inondazione")
        send_email_alert(email_recipient, alert_count)
    else:
        print("Nessuna zona a rischio significativa")
        
    plt.figure(figsize=(8, 6))
    plt.title("Zone a rischio alluvione")
    plt.imshow(risk_zones, cmap='Reds')
    plt.colorbar()
    plt.show()
    
    return risk_zones

def send_email_alert(email_recipient, alert_count):
    sender_email = ""
    sender_password = ""
    subject = "Allerta inondazione"
    body = f"Attenzione, sono state identificate {alert_count} zone a rischio inondazione"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = email_recipient
    
    try:
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email_recipient, msg.as_string())
        print("Email di allerta inviata con successo.")
    except Exception as e:
        print("Errore nell'invio dell'email:", e)