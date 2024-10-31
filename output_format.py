# award class to store information about each award
class award:
    def __init__(self, name):
        self.name = name
        self.winner = {}
        self.nominees = {}
        self.presenters = {}
        self.votes = 1
    
    # add a new person associted with the award
    def new_person(self, name, role):
        if role == 'winner':
            self.winner.update({name:1})
        elif role == 'nominee':
            self.nominees.update({name:1})
        elif role == 'presenter':
            self.presenters.update({name:1})
    
    # a repeat-appearence of a person in a role gives them an additional vote for that role (more votes = more likely)
    def person_vote(self, name, role):
        if role == 'winner':
            self.winner[name] += 1
        elif role == 'nominee':
            self.nominees[name] += 1
        elif role == 'presenter':
            self.presenters[name] += 1
    
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
awards = {}

# Split up each row of our dataframe and apply appropriate function to update our answers
def extract_answers(text):
    nominee, curr_award, role = text.split(' | ')
    
    if curr_award not in awards:
        awards.update({curr_award:award(curr_award)})
    else:
        awards[curr_award].award_vote()
        
    if not awards[curr_award].contains(nominee, role):
        awards[curr_award].new_person(nominee, role)
    else:
        awards[curr_award].person_vote(nominee, role)

### NEED TO JSONIFY OUTPUT
