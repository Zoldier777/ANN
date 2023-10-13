from ChatBotTrainer import *
import Questions

#load saved states of trained network


dirname = os.path.dirname(os.path.abspath("ANN"))
filename = os.path.join(dirname, 'network.pth')
network = torch.load(filename)

dirname = os.path.dirname(os.path.abspath("ANN"))
filename = os.path.join(dirname, 'word_vectorizer.pickle')
word_vectorizer = pickle.load(open(filename, "rb"))

labels = {
    0:  'empty',
    1:  'sadness',
    2:  'enthusiasm',
    3:  'neutral',
    4:  'worry',
    5:  'love',
    6:  'fun',
    7:  'surprise',
    8:  'hate',
    9:  'happiness',
    10: 'boredom',
    11: 'relief',
    12: 'anger'
}

questions = Questions.Questions("questions.txt")
question_rnd = questions.random # try questions.random_rm() to show every question only ones (Then don't loop more then number of questions times)

loop = True
question = question_rnd()
while (loop == True):
    print(question)
    x = input()#
    #preprocessing of reply needs to added, i suppose
    validation_data = word_vectorizer.transform([x])
    validation_data = validation_data.todense()
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    prediction= network(validation_x_tensor)
    guess = torch.argmax(prediction, dim=-1)

    # Example of how the labels dict can be used
    #print("This sentence has the feeling of: " + labels[guess.item()])


    #logic for handling the bots' reply
    #print(prediction) #to check if statement below is correct



    if ((guess.item()==0 and prediction[0][0]>0.8) or (guess.item()==1 and prediction[0][1]>0.8)):  #the network be more than 80% about its reply otherwise the user need to elaborate
        if(guess.item()==0):
            print("BOT:Im very sorry to hear that, we will take it into consideration")
        else:
            print("BOT:Glad to Hear that")

        print("BOT:Anything else? Y/N")
        x = input()
        if(x=="N" or x== "n"):
            loop = False

        question = question_rnd()
    else:
        print("BOT:I did not understand, Try to be more specific")