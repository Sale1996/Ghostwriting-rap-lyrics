# Ghostwriting-rap-lyric

Project which generates rap lyrics in style of certain rapper. 

Structure of repo:
1. data - contains theCMUDict dictionary files, scrapped artist links,  artist plain song lyrics texts and artist formated song lyrics texts.
2. scrape_and_form_lyrics_data - contains python scripts from scrapping artist lyrics to formate them properly and convert to phonetic form.
3. g2p_cmudict - contains python representation of grapheme to phoneme LSTM Encoder-Decoder model. Credits to: - fehiepsi
4. lyrics_generator - contains pyhton representation of bidirectional LSTM model for generating rap lyrics.

