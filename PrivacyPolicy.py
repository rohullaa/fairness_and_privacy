import pandas as pd
import numpy as np


class PrivacyPolicy:
    def __init__(self, features, actions, outcome):
        self.features = features
        self.actions = actions
        self.outcome = outcome

    def exponential_mechanism(self, epsilon):
        """Given a set of actions and a utility function, this function returns the 'best' action."""
        best_actions = []
        for i in range(self.features.shape[0]):
            utility = np.array([self.get_utility(self.features.iloc[i,:], action, self.outcome.iloc[i,:]) for action in self.actions])
            policy_probs = exp(epsilon*utility/2*self.sensitivity)
            policy_probs = policy_probs/np.linalg.normalize(policy_probs, ord=1)
            best_actions.append(np.random.choice(self.actions.columns, 1, p=policy_probs)[0])
        return best_actions

    def get_utility(self, features, action, outcome):
        utility = 0
        utility -= 0.2 * sum(outcome[:,symptom_names['Covid-Positive']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Taste']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Fever']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Headache']])
        utility -= 0.5 * sum(outcome[:,symptom_names['Pneumonia']])
        utility -= 0.2 * sum(outcome[:,symptom_names['Stomach']])
        utility -= 0.5 * sum(outcome[:,symptom_names['Myocarditis']])
        utility -= 1.0 * sum(outcome[:,symptom_names['Blood-Clots']])
        utility -= 100.0 * sum(outcome[:,symptom_names['Death']])
        utility += 100.0 * actions[actions['Action3']]

        self.sensitivity = 100
        return utility

symptoms = pd.DataFrame({
'Covid-Postive': [1,1,1,1,1,1,1,1,1,1],
'Taste': [0,0,0,0,0,1,1,1,1,1],
'Fever': [1,0,1,0,1,0,1,0,1,0],
'Headace': [1,1,0,0,1,1,0,0,1,1],
'Pnemonua': [0,0,1,1,0,0,1,1,0,0],
'Stomach': [0,1,0,1,0,1,0,1,0,1],
'Myocarditis': [0,0,0,0,0,0,0,0,0,0],
'Blood-Clots': [0,0,0,0,0,0,0,0,0,0],
'Death': [0,0,0,0,0,0,0,0,0,1]})
actions = pd.Dataframe({
'Action0': [1,1,1,0,0,0,0,0,0,0],
'Action1': [0,0,0,1,1,1,1,0,0,0],
'Action2': [0,0,0,0,0,1,1,1,1,0],
'Action3': [0,0,0,0,0,1,0,0,0,0]})
outcome = pd.DataFrame({
'Covid-Postive': [1,1,1,0,1,1,1,0,1,1],
'Taste': [0,0,0,0,0,1,0,0,1,1],
'Fever': [1,0,0,0,1,0,0,0,1,0],
'Headace': [1,1,0,0,1,1,0,0,1,1],
'Pnemonua': [0,0,1,0,0,0,1,0,0,0],
'Stomach': [0,0,0,1,0,0,0,1,0,1],
'Myocarditis': [0,0,0,0,0,0,0,0,0,0],
'Blood-Clots': [0,0,0,0,0,0,0,0,0,0],
'Death': [0,0,0,0,0,0,0,0,0,1]})

privpol = PrivacyPolicy(symptoms, actions, outcome)
privpol.exponential_mechanism(0.1)
