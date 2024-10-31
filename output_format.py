import pandas as pd

# award class to store information about each award
class award:
    def __init__(self, name):
        self.name = name
        self.winner = {}
        self.nominees = {}
        self.presenters = {}
        self.votes = 1
    
    # add a new person associted with the award
    def new_person(self, name, role, pcount=1):
        if role == 'winner':
            self.winner.update({name:pcount})
        elif role == 'nominee':
            self.nominees.update({name:pcount})
        elif role == 'presenter':
            self.presenters.update({name:pcount})
        
        if pcount > 1:
            self.votes += pcount - 1
    
    # a repeat-appearence of a person in a role gives them an additional vote for that role (more votes = more likely)
    def person_vote(self, name, role):
        if role == 'winner':
            self.winner[name] += 1
        elif role == 'nominee':
            self.nominees[name] += 1
        elif role == 'presenter':
            self.presenters[name] += 1
    
    def remove_person(self, name, role):
        if role == 'winner':
            pcount = self.winner.pop(name)
            self.votes -= pcount
        elif role == 'nominee':
            pcount = self.nominees.pop(name)
            self.votes -= pcount
        elif role == 'presenter':
            pcount = self.presenters.pop(name)
            self.votes -= pcount
        
        return name, role, pcount
    
    # a repeat award gets more votes (more votes = more likely)
    def award_vote(self):
        self.votes += 1
    
    # check if a person is already present in a particular award
    def contains(self, name, role):
        if role == 'winner':
            return name in self.winner
        elif role == 'nominee':
            return name in self.nominees
        elif role == 'presenter':
            return name in self.presenters
    
    def output(self):
        return {self.name:
                { 
                    "nominees": [(n, self.nominees[n]) for n in self.nominees],
                    "presenters":[(p, self.presenters[p]) for p in self.presenters], 
                    "winner":[(w, self.winner[w]) for w in self.winner],
                    "votes":self.votes
                }
        }


# list of awards and their respective winners, nominees, and presenters, along with the number of times they appear in our data
awards_list = {}

# Split up each row of our dataframe and apply appropriate function to update our answers
def extract_answers(text):
    nominee, curr_award, role = text.split(' | ')
    nominee = nominee.lower()
    curr_award = curr_award.lower()
    
    if curr_award not in awards_list:
        awards_list.update({curr_award:award(curr_award)})
    else:
        awards_list[curr_award].award_vote()
        
    if not awards_list[curr_award].contains(nominee, role):
        awards_list[curr_award].new_person(nominee, role)
    else:
        awards_list[curr_award].person_vote(nominee, role)

# function to move data from one award to another
def move_data(a1: award, a2:award):
    win_temp = a1.winner.keys()
    nom_temp = a1.nominees.keys()
    pres_temp = a1.presenters.keys()
    for w in win_temp:
        if a2.contains(w, "winner"):
            name, role, pcount = a1.remove_person(w, "winner")
            a2.new_person(name, role, pcount)
    for n in nom_temp:
        if a2.contains(n, "nominee"):
            name, role, pcount = a1.remove_person(n, "nominee")
            a2.new_person(name, role, pcount)
    for p in pres_temp:
        if a2.contains(p, "presenters"):
            name, role, pcount = a1.remove_person(p, "presenters")
            a2.new_person(name, role, pcount)


### NEED TO AGGREGATE AWARD DATA

### NEED TO JSONIFY OUTPUT


if __name__ == "__main__":
    answers = pd.read_csv('winners_and_awards.csv')['text']
    answers = answers.dropna()
    
    answers.apply(extract_answers)
    
    top_awards = dict(sorted(awards_list.items(), key=lambda item: item[1].votes, reverse=True))
    for i in top_awards:
        print(top_awards[i].output())
        
    
    