from exp_manager import ExpManeger
import json

config = 'plot.json'


i = ExpManeger(config)
#i.DrawBestSkillScore('correlation')
#i.GetBestSkillScore('correlation', 'h')
# i.DrawLossAll()
i.OverAllEval()
# i.DrawSkillScoreAll()
