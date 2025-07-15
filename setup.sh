#!bin/bash
pip install -r requirements.txt

mkdir -p dataset

# download laws  https://drive.google.com/file/d/1AjyK6SrLsP_UmCzOZCzPEDDmOMVwzcW8/view?usp=drive_link
gdown  1AjyK6SrLsP_UmCzOZCzPEDDmOMVwzcW8 -O dataset/alqac25_law.json

# download question https://drive.google.com/file/d/1hYYR1olfJikNOO0j3-RMTwKoO7uLJiVO/view?usp=drive_link
gdown  1hYYR1olfJikNOO0j3-RMTwKoO7uLJiVO -O dataset/alqac25_train.json

# download wseg laws https://drive.google.com/file/d/1vGklxoVVVa2qdGtLdxRA7FOopJnMF0E4/view?usp=sharing
gdown 1vGklxoVVVa2qdGtLdxRA7FOopJnMF0E4 -O dataset/wseg_alqac25_law.json 

# download wseg questions https://drive.google.com/file/d/1iErE-oaclXExnurBV0-HPkjdSJbyE5Mq/view?usp=sharing
gdown 1iErE-oaclXExnurBV0-HPkjdSJbyE5Mq -O dataset/wseg_alqac25_train.json