#This script downloads a sample of the Twitter firehose to an SQLite database

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import json
import dataset

TRACK_TERMS = ["#trump", "#obama", "#clinton", "#bush"]
LANGUAGES = ['en']
CONNECTION_STRING = "sqlite:///TweetData.db"

TABLE_NAME = "sample1"
TABLE_NAME2 = "sample2"
 
#insert your info here (obtained with a Twitter dev account)
consumer_key = 'XXX'
consumer_secret = 'XXX'
access_token = 'XXX'
access_secret = 'XXX'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

class StreamListener(StreamListener):

    def on_status(self, status):
        db = dataset.connect(CONNECTION_STRING)
        try: 
        
            #original Tweets saved to TABLE_NAME, Retweets saved to TABLE_NAME2
            if not hasattr(status,'retweeted_status'): 
                table = db[TABLE_NAME]
                
                if coords is not None:
                    coords = json.dumps(status.coordinates)#convert coordinates to string
                
                 #Put all these variables into table
                table.insert(dict(
                        user_description=status.user.description,
                        user_location=status.user.location,
                        coordinates=coords,
                        text=status.text,
                        user_name=status.user.screen_name,
                        user_created=status.user.created_at,
                        user_followers=status.user.followers_count,
                        id_str=status.id_str,
                        created=status.created_at,
                        retweet_count=status.retweet_count,
                        user_bg_color=status.user.profile_background_color,
                        ))
                
            else: 
                table = db[TABLE_NAME2]
                
                if coords is not None:
                    coords = json.dumps(status.coordinates)#convert coordinates to string
                
                table.insert(dict(
                        user_description=status.user.description,
                        user_location=status.user.location,
                        coordinates=coords,
                        text=status.text,
                        user_name=status.user.screen_name,
                        user_created=status.user.created_at,
                        user_followers=status.user.followers_count,
                        id_str=status.id_str,
                        created=status.created_at,
                        retweet_count=status.retweet_count,
                        user_bg_color=status.user.profile_background_color,
                        ))
        
        except:
            print('This one returned nothing')

      
    def on_error(self, status_code):
        if status_code == 420:
            return False
 
twitter_stream = Stream(auth, StreamListener())
twitter_stream.filter(track=TRACK_TERMS, languages=LANGUAGES)
twitter_stream.sample(languages = ['en','nl'])

#This script runs indefinitely, time parameter still needs to be added
