'''Version 0.4'''
from helper_functions import *

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    hosts = help_get_hosts()

    return hosts

def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    award_list = help_get_awards()
    
    return award_list

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    award_names = [
        "best screenplay - motion picture",
        "best director - motion picture",
        "best performance by an actress in a television series - comedy or musical",
        "best foreign language film",
        "best performance by an actor in a supporting role in a motion picture",
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
        "best motion picture - comedy or musical",
        "best performance by an actress in a motion picture - comedy or musical",
        "best mini-series or motion picture made for television",
        "best original score - motion picture",
        "best performance by an actress in a television series - drama",
        "best performance by an actress in a motion picture - drama",
        "cecil b. demille award",
        "best performance by an actor in a motion picture - comedy or musical",
        "best motion picture - drama",
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
        "best performance by an actress in a supporting role in a motion picture",
        "best television series - drama",
        "best performance by an actor in a mini-series or motion picture made for television",
        "best performance by an actress in a mini-series or motion picture made for television",
        "best animated feature film",
        "best original song - motion picture",
        "best performance by an actor in a motion picture - drama",
        "best television series - comedy or musical",
        "best performance by an actor in a television series - drama",
        "best performance by an actor in a television series - comedy or musical"
    ]
    nom_output = help_get_nominees()
    
    nominees = convert_results_to_match_awards(award_names, nom_output)

    return nominees

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    award_names = [
        "best screenplay - motion picture",
        "best director - motion picture",
        "best performance by an actress in a television series - comedy or musical",
        "best foreign language film",
        "best performance by an actor in a supporting role in a motion picture",
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
        "best motion picture - comedy or musical",
        "best performance by an actress in a motion picture - comedy or musical",
        "best mini-series or motion picture made for television",
        "best original score - motion picture",
        "best performance by an actress in a television series - drama",
        "best performance by an actress in a motion picture - drama",
        "cecil b. demille award",
        "best performance by an actor in a motion picture - comedy or musical",
        "best motion picture - drama",
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
        "best performance by an actress in a supporting role in a motion picture",
        "best television series - drama",
        "best performance by an actor in a mini-series or motion picture made for television",
        "best performance by an actress in a mini-series or motion picture made for television",
        "best animated feature film",
        "best original song - motion picture",
        "best performance by an actor in a motion picture - drama",
        "best television series - comedy or musical",
        "best performance by an actor in a television series - drama",
        "best performance by an actor in a television series - comedy or musical"
    ]
    win_output = help_get_winners()
    
    winners = convert_results_to_match_awards(award_names, win_output)

    return winners

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    award_names = [
        "best screenplay - motion picture",
        "best director - motion picture",
        "best performance by an actress in a television series - comedy or musical",
        "best foreign language film",
        "best performance by an actor in a supporting role in a motion picture",
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
        "best motion picture - comedy or musical",
        "best performance by an actress in a motion picture - comedy or musical",
        "best mini-series or motion picture made for television",
        "best original score - motion picture",
        "best performance by an actress in a television series - drama",
        "best performance by an actress in a motion picture - drama",
        "cecil b. demille award",
        "best performance by an actor in a motion picture - comedy or musical",
        "best motion picture - drama",
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
        "best performance by an actress in a supporting role in a motion picture",
        "best television series - drama",
        "best performance by an actor in a mini-series or motion picture made for television",
        "best performance by an actress in a mini-series or motion picture made for television",
        "best animated feature film",
        "best original song - motion picture",
        "best performance by an actor in a motion picture - drama",
        "best television series - comedy or musical",
        "best performance by an actor in a television series - drama",
        "best performance by an actor in a television series - comedy or musical"
    ]

    presenters_output = help_get_presenters()
    
    presenters = convert_results_to_match_awards(award_names, presenters_output)
    return presenters

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    print("Pre-ceremony processing complete.")
    return

def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    year = 2013
    award_names = [
        "best screenplay - motion picture",
        "best director - motion picture",
        "best performance by an actress in a television series - comedy or musical",
        "best foreign language film",
        "best performance by an actor in a supporting role in a motion picture",
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
        "best motion picture - comedy or musical",
        "best performance by an actress in a motion picture - comedy or musical",
        "best mini-series or motion picture made for television",
        "best original score - motion picture",
        "best performance by an actress in a television series - drama",
        "best performance by an actress in a motion picture - drama",
        "cecil b. demille award",
        "best performance by an actor in a motion picture - comedy or musical",
        "best motion picture - drama",
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
        "best performance by an actress in a supporting role in a motion picture",
        "best television series - drama",
        "best performance by an actor in a mini-series or motion picture made for television",
        "best performance by an actress in a mini-series or motion picture made for television",
        "best animated feature film",
        "best original song - motion picture",
        "best performance by an actor in a motion picture - drama",
        "best television series - comedy or musical",
        "best performance by an actor in a television series - drama",
        "best performance by an actor in a television series - comedy or musical"
    ]

    additional_info = get_best_dressed_and_jokes('text_cleaned.csv')
    best_dressed = additional_info["best_dressed"]
    best_joke = additional_info["best_joke"]

    cleaned_data = clean_data()
    # human_readable_version(award_names)
    # winners = get_winner(year)
    # presenters = get_presenters(year)
    # hosts = get_hosts(year)
    # print(get_nominees(year))
    # print("Best Dressed:", best_dressed)
    # print("Best Joke:", best_joke)
    return

if __name__ == '__main__':
    main()