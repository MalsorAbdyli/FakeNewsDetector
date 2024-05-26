import pickle

var = input("Please enter the news text you want to verify: ")
print("You entered: " + var)

def detecting_fake_news(var):    
    load_model = pickle.load(open('model.pkl', 'rb'))
    tfidf_v = TfidfVectorizer()
    vectorized_input_data = tfidf_v.transform([var])
    prediction = load_model.predict(vectorized_input_data)
    prob = load_model.predict_proba(vectorized_input_data)

    print("The given statement is", "Fake" if prediction[0] == 0 else "Real")
    print("The truth probability score is", prob[0][1])

if __name__ == '__main__':
    detecting_fake_news(var)
