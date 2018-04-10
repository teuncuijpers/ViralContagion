# Insert your key and token, obtained from a Twitter dev account, and this script should be ready to go. 

import tweepy
import dataset

consumer_key = 'XXX'
consumer_secret = 'XXX'
access_token = 'XXX'
access_secret = 'XXX'

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret) #This handler can handle 450 requests per 15min

api = tweepy.API(auth, wait_on_rate_limit=True,
				   wait_on_rate_limit_notify=True) #Gives warning when limit is reached and continues when 15min are over

if (not api):
    print ("Can't Authenticate")  

QUERY = '#trump OR #clinton'
MAX_TWEETS = 200000 #max amount of Tweets to be downloaded
MAX_PER_QUERY = 100 #times 450 queries is 45000 Tweets per 15 minutes
LANGUAGES = ['en']
CONNECTION_STRING = "sqlite:///Data.db"
TABLE_NAME = "sample1"

COUNT_100 =[]
for i in range(0,100000,10):
    COUNT_100.append(i)

try:
    for counter, status in enumerate(tweepy.Cursor(api.search,q=QUERY, count=MAX_PER_QUERY).items(MAX_TWEETS)):
        if not hasattr(status,'retweeted_status'): #Only print Tweets that are not Retweets
        
            db = dataset.connect(CONNECTION_STRING) #Open SQLite db
            
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
                    
            if counter in COUNT_100:
                print(counter)

    print('SUCCESS!!')
    
except: 
    print('ERROR')

