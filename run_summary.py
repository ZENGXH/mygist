from exp_manager import ExpManeger
import json

config = 'plot.json'


i = ExpManeger(config)
i.GetBestSkillScore('correlation', 'h')
i.DrawLossAll()

i.DrawSkillScoreAll()
