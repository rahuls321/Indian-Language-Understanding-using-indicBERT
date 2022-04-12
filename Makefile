install:
	pip install -r requirements.txt
	wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HgCvNPg6UyKkwb8T3Y09xujBZLS-SVqz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HgCvNPg6UyKkwb8T3Y09xujBZLS-SVqz" -O ./utils/glove_vec.pickle && rm -rf /tmp/cookies.txt
run:
	./run.sh
