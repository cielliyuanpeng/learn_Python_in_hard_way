class Song(object):
    def __init__(self,lyric):
        self.lyric = lyric
        pass
    def sing_me_a_song(self):
        for line in self.lyric:
            print(line)
            pass
        pass
    pass

happy_bday = Song(["happy b day 2 U",
                    "I dont want to get sued",
                    "So i 'll stop right here"])

bulls_on_parade = Song(["They really around tha family",
                        "With pockets full of shells"])

happy_bday.sing_me_a_song()
bulls_on_parade.sing_me_a_song()
