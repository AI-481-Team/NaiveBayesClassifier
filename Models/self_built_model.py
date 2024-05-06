import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class SelfBuiltNB_V1:
    """
    V1 uses attempt 1 from playground notebook to calculate the conditional probabilities
    """
    def fit(self, mails, labels):
        self.prior_probas =  NB_Utils.get_prior_probas(labels)
        self.conditional_probas = NB_Utils.get_cond_probas_attempt_1(p_mails=mails,p_labels=labels)
       
    def predict(self,mails):
        results = [NB_Utils.classify_message(msg, self.conditional_probas, self.prior_probas) for msg in mails]
        return results


class SelfBuiltNB_V2:
    """
    V2 uses attampt 2 from playground notebook to calculate the conditional probabilities
    """
    def fit(self, mails, labels):
        self.prior_probas =  NB_Utils.get_prior_probas(labels)
        self.conditional_probas = NB_Utils.get_cond_probas_attempt_2(p_mails=mails, p_labels=labels)
       
    def predict(self,mails):
        results = [NB_Utils.classify_message(msg, self.conditional_probas, self.prior_probas) for msg in mails]
        return results


class NB_Utils:
    @staticmethod
    def get_prior_probas(labels:pd.DataFrame):

        spam_count_total = np.sum(labels == 1)
        not_spam_count_total = np.sum(labels == 0)
        total = spam_count_total + not_spam_count_total

        P_spam = spam_count_total / total
        P_not_spam = not_spam_count_total / total

        ret_dict ={
            "P_spam" : P_spam, 
            "P_not_spam": P_not_spam
        } 
        return ret_dict


    @staticmethod
    def classify_message(message, cond_probas, prior_probas):

        cv = CountVectorizer(vocabulary=cond_probas['Word'].values)

        #with this CountVectorizer you can make a vector from the message
        #same as the word_count_matrix, but only for one mail text
        message_vector = cv.transform([message]).toarray().flatten()

        #for the calculation it's somehow better to use log of the actual probabs
        #avoids some kind of problem that arieses if one calculates with very small numbers
        log_prob_spam = np.log(prior_probas['P_spam'])
        log_prob_not_spam = np.log(prior_probas['P_not_spam'])

        #in this loop the calculation for the vector is done for spam=yes as well as for spam=no
        for word, count in zip(cv.get_feature_names(), message_vector): 
            if count > 0:  
                if word in cond_probas['Word'].values: #could be that test data contains words that haavent been in training data
                    word_data = cond_probas[cond_probas['Word'] == word] # get the word's conditional probabilities
                    log_prob_spam += count * np.log(word_data['P(Word|Spam)'].values[0]) 
                    log_prob_not_spam += count * np.log(word_data['P(Word|NotSpam)'].values[0])

        #make the choice by comparing the two values
        if log_prob_spam > log_prob_not_spam:
            return 1
        else:
            return 0
        

    @staticmethod
    def get_cond_probas_attempt_1(p_mails:pd.DataFrame, p_labels:pd.DataFrame):
        p_mails = p_mails.reset_index(drop=True)
        p_labels = p_labels.reset_index(drop=True)
        

        cv = CountVectorizer()
        word_count_matrix = cv.fit_transform(p_mails.to_numpy()).toarray()

        spam_count_total = np.sum(p_labels == 1)
        not_spam_count_total = np.sum(p_labels == 0)
        
        unique_word_count = len(word_count_matrix[0]) #how many elements does one row have + eunique words in corpus

        words = cv.get_feature_names()
        word_proba_array = np.zeros((2,unique_word_count))
        
        #Get the idices for all spam rows and for all non spam rows
        spam_rows_indices = p_labels[p_labels == 1].index.tolist()
        not_spam_rows_indices = p_labels[p_labels == 0].index.tolist()

        # Calculate the two conditional probabilities for each word
        for word_idx in range(len(word_count_matrix[0])):
            
            #calculating for spam=yes rows
            for row_index in spam_rows_indices:
                word_count = word_count_matrix[row_index][word_idx]
                if(word_count > 0): #only checks if the word appears, not how often
                    word_proba_array[0][word_idx] += 1 / (spam_count_total)

            #calculating for spam=no rows
            for row_index in not_spam_rows_indices:
                word_count = word_count_matrix[row_index][word_idx]
                if(word_count > 0): #only checks if the word appears, not how often
                    word_proba_array[1][word_idx] += 1 / (not_spam_count_total)
                    #-> same as before but for spam=no mails

        probabilities_df = pd.DataFrame({
            'Word': words,
            'P(Word|Spam)': word_proba_array[0],
            'P(Word|NotSpam)': word_proba_array[1]
        })

        return probabilities_df

    @staticmethod
    def get_cond_probas_attempt_2(p_mails:pd.DataFrame, p_labels:pd.DataFrame):

        cv = CountVectorizer()
        word_count_matrix = cv.fit_transform(p_mails.to_numpy()).toarray()

        alpha = 1
        unique_word_count = len(cv.get_feature_names())
        words = np.array(cv.get_feature_names())

        #instead of looping through the whole dataset counting word appearcances and adding probabilities, make 
        #an array that counts each word's appearances over the whole dataset (also considers how often a word
        # appears in a sentence and not only if or if it doean's appear as in the 1.attempt)
        spam_word_counts = word_count_matrix[p_labels == 1].sum(axis=0)
        not_spam_word_counts = word_count_matrix[p_labels == 0].sum(axis=0)

        #Calculating the contiditional probalbilities by going  spam_word_counts/ total amount of spam word
        # + it adds laplace smoothing as seen here:  https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece
        word_proba_spam = (spam_word_counts + alpha) / (spam_word_counts.sum() + alpha * unique_word_count)
        word_proba_not_spam = (not_spam_word_counts + alpha) / (not_spam_word_counts.sum() + alpha * unique_word_count)

        probabilities_df = pd.DataFrame({
            'Word': words,
            'P(Word|Spam)': word_proba_spam,
            'P(Word|NotSpam)': word_proba_not_spam
        })

        return probabilities_df

    #Method for classifying a single message
    @staticmethod
    def classify_message(message, cond_probas, prior_probas):
        

        #the next line make a countvectorizer that uses the same 
        #columns names as in the training (column names = unique words)
        #this is done by adding the same vocabulary as in Word
        cv = CountVectorizer(vocabulary=cond_probas['Word'].values)

        #with this CountVectorizer you can make a vector from the message
        #same as the word_count_matrix, but only for one mail text
        message_vector = cv.transform([message]).toarray().flatten()

        #for the calculation it's somehow better to use log of the actual probabs
        #avoids some kind of problem that arieses if one calculates with very small numbers
        log_prob_spam = np.log(prior_probas['P_spam'])
        log_prob_not_spam = np.log(prior_probas['P_not_spam'])

        #in this loop the calculation for the vector is done for spam=yes as well as for spam=no
        for word, count in zip(cv.get_feature_names(), message_vector): 
            if count > 0:  
                if word in cond_probas['Word'].values: #could be that test data contains words that haavent been in training data
                    word_data = cond_probas[cond_probas['Word'] == word] # get the word's conditional probabilities
                    log_prob_spam += count * np.log(word_data['P(Word|Spam)'].values[0]) 
                    log_prob_not_spam += count * np.log(word_data['P(Word|NotSpam)'].values[0])

        #make the choice by comparing the two values
        if log_prob_spam > log_prob_not_spam:
            return 1
        else:
            return 0

