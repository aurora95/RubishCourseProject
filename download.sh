#! /bin/bash

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install python-pip
sudo pip install kaggle-cli
sudo pip install -U kaggle-cli
read -p 'Username: ' uservar
read -sp 'Password: ' passvar
echo ''
competition='plant-seedlings-classification'
kg config -u $uservar -p $passvar -c $competition
mkdir ~/kaggle
cd ~/kaggle
echo 'Downloading Kaggle Competition Data...'
kg download
sudo apt-get install unzip -y
echo 'Unzipping Training Data...'
unzip train.zip
echo 'Successfully unizpped the training data!'
echo 'Unzipping the Test Data...'
unzip test.zip
echo 'Successfully unizpped the test data!'
echo 'Unzipping the sample submission...'
unzip sample_submission.csv.zip
rm sample_submission.csv.zip test.zip train.zip
