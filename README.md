# Chatbot_Web_Application

## Steps to run the application:

1. Clone the github repository and cd into the Chatbot folder. Run the following commands
2. Download the pickle file in the drive link to chatbot folder.
3. docker build -t imagename .
4. docker run -p 4000:80 imagename
5. You can access the chatbot at localhost:4000

## Custom LLM Knowledge Base

I have developed a Custom LLM, trained on a document on PAN Card Applications. The document can be found inside the repo. 
Filename: KnowledgeDocument(pan_card_services).txt

### Few Example Questions that can be asked

1. What is the Cost of new PAN card?
2. What are the Charges for reprinting the PAN Card?
3. What is Form 49aa?
4. Is it mandatory to link Aadhaar with PAN for NRI?
