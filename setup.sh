#!bin/bash
pip install -r requirements.txt

mkdir -p dataset

# download laws  https://drive.google.com/file/d/1AjyK6SrLsP_UmCzOZCzPEDDmOMVwzcW8/view?usp=drive_link
gdown  1AjyK6SrLsP_UmCzOZCzPEDDmOMVwzcW8 -O dataset/alqac25_law.json

# download question https://drive.google.com/file/d/1hYYR1olfJikNOO0j3-RMTwKoO7uLJiVO/view?usp=drive_link
gdown  1hYYR1olfJikNOO0j3-RMTwKoO7uLJiVO -O dataset/alqac25_train.json
