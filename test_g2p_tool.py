from g2p_en import G2p

texts = ["<verse_start>", # number -> spell-out
         "You never met me, and you'll probably never see me again<end_line>", # e.g. -> for example
         "but I know you - the name's Slim - you want revenge?<end_line>", # homograph
         "Then don't shoot, I'm in the same boots as you<end_line>",
         "I'm tellin the truth, I got a price on my head too, cause when you..<end_line>",
         "<verse_end>"] # newly coined word
g2p = G2p()
for text in texts:
    out = g2p(text)
    print(out)