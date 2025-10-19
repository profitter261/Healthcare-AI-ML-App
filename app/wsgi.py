from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

# Import all the Flask apps
from Home_Page.app import app as home_app
from LOS_prediction.LOS import app as los_app
from Disease_prediction.Disease import app as disease_app
from Patient_readmission_and_detoriation.patient import app as patient_app
from Image_Diagnostics.app import app as image_app
from Patient_discharge_summarizer.summarizer import app as summarizer_app
from Drug_sentiment_analysis.Drug import app as senti_app
from Patient_clustering.patient_cluster import app as cluster_app
from Chatbot.chatbot import app as chat_app

# Mount the sub-apps
application = DispatcherMiddleware(home_app, {
    '/LOS': los_app,
    '/Disease': disease_app,
    '/Patient': patient_app,
    '/Image-Diagnostics': image_app,
    '/Summarizer': summarizer_app,
    '/Senti': senti_app,
    '/Cluster': cluster_app,
    '/chat': chat_app
})

# Run combined app (for local development)
if __name__ == "__main__":
    run_simple('0.0.0.0', 5000, application, use_reloader=False, use_debugger=True)
