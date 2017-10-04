
# coding: utf-8
# Script extracts records for the specified city

import argparse
import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
    
	parser = argparse.ArgumentParser(
	    description='Extract city data from the Yelp Datsets.',
	    )

	parser.add_argument(
	    'city',
	    type=str,
	    help='The city name.',
	    )

	args = parser.parse_args()

	#The data directory
	data_dir = '../data/csv'


	print('Read the business dataset')
	data_business = pd.read_csv(data_dir + '/yelp_academic_dataset_business.csv')

	print('Read the review dataset')
	data_review = pd.read_csv(data_dir + '/yelp_academic_dataset_review.csv')

	print('Read the tip dataset')
	data_tip = pd.read_csv(data_dir + '/yelp_academic_dataset_tip.csv')

	print('Read the user dataset')
	data_user = pd.read_csv(data_dir + '/yelp_academic_dataset_user.csv')

	print('Read the checkin dataset')
	data_checkin = pd.read_csv(data_dir + '/yelp_academic_dataset_checkin.csv')

	#set the city
	city = args.city

	print('Get the businesses for the specified city')
	city_business = data_business[data_business['city']==city]

	print('Get the reviews of the businesses for the specified city')
	city_review = data_review[data_review['business_id'].isin(city_business['business_id'])]

	print('Get the tips of the businesses for the specified city')
	city_tip = data_tip[data_tip['business_id'].isin(city_business['business_id'])]

	print('Get the checkins of the businesses for the specified city')
	city_checkin = data_checkin[data_checkin['business_id'].isin(city_business['business_id'])]

	print('Get the users who posted reviews')
	city_user = data_user[data_user['user_id'].isin(city_review['user_id'].drop_duplicates())]

	#The directory where the files for the particular city are created
	city_dir = '../data/city/' + city

	#create the directory
	if not os.path.exists(city_dir):
		os.makedirs(city_dir)

	print('write the city business data')
	city_business.to_csv(os.path.join(city_dir, 'yelp_academic_dataset_business.csv'))

	print('write the reviews')
	city_review.to_csv(os.path.join(city_dir, 'yelp_academic_dataset_review.csv'))

	print('write the tips')
	city_tip.to_csv(os.path.join(city_dir, 'yelp_academic_dataset_tip.csv'))

	print('write the users')
	city_user.to_csv(os.path.join(city_dir, 'yelp_academic_dataset_user.csv'))

	print('write the checkins')
	city_checkin.to_csv(os.path.join(city_dir, 'yelp_academic_dataset_checkin.csv'))
