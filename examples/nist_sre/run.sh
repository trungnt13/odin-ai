# For training: fisher,mx6,sre04,sre05,sre06,sre08,sre10,swb,voxceleb1,voxceleb2
# For noise   : musan, rirs

# Test training on voxceleb1
python train_xvec.py mfcc_musan_rirs -exclude fisher,mx6,sre04,sre05,sre06,sre08,sre10,swb,voxceleb2 -mindur 1 -minutt 8 --override
# Test training on voxceleb2
python train_xvec.py mfcc_musan_rirs -exclude fisher,mx6,sre04,sre05,sre06,sre08,sre10,swb,voxceleb1 -mindur 1 -minutt 8 --override
# all sre and mixer
python train_xvec.py mfcc_musan_rirs -exclude fisher,swb,voxceleb1,voxceleb2 -mindur 4 -minutt 8 --override
# only fisher
python train_xvec.py mfcc_musan_rirs -exclude mx6,sre04,sre05,sre06,sre08,sre10,swb,voxceleb1,voxceleb2 -mindur 1 -minutt 8 --override
# everything except voxceleb
python train_xvec.py mfcc_musan_rirs -exclude voxceleb1,voxceleb2 -mindur 3 -minutt 8 --override

# test training on only one of the sre (without noise)
# sre04
python train_xvec.py mfcc_musan_rirs -exclude fisher,mx6,sre05,sre06,sre08,sre10,swb,voxceleb1,voxceleb2,noise
# sre05
python train_xvec.py mfcc_musan_rirs -exclude fisher,mx6,sre04,sre06,sre08,sre10,swb,voxceleb1,voxceleb2,noise
# sre06
python train_xvec.py mfcc_musan_rirs -exclude fisher,mx6,sre04,sre05,sre08,sre10,swb,voxceleb1,voxceleb2,noise
# sre08
python train_xvec.py mfcc_musan_rirs -exclude fisher,mx6,sre04,sre05,sre06,sre10,swb,voxceleb1,voxceleb2
# sre10
python train_xvec.py mfcc_musan_rirs -exclude fisher,mx6,sre04,sre05,sre06,sre08,swb,voxceleb1,voxceleb2,noise

# swb
python train_xvec.py mfcc_musan_rirs -exclude fisher,mx6,sre04,sre05,sre06,sre08,sre10,voxceleb1,voxceleb2,noise

# test training on only sre06 and noise
python train_xvec.py mfcc_musan_rirs -exclude fisher,mx6,sre04,sre05,sre08,sre10,swb,voxceleb1,voxceleb2

# No Voxceleb datasets
python train_xvec.py mfcc_musan_rirs -exclude voxceleb1,voxceleb2
python make_score.py mfcc_musan_rirs -sys xvec -sysid -1 -score sre18dev,sre18eval -backend sre04,sre05,sre06,sre08,sre10 -exclude voxceleb1,voxceleb2

# same but without noise
python train_xvec.py mfcc_musan_rirs -exclude voxceleb1,voxceleb2,noise
python make_score.py mfcc_musan_rirs -sys xvec -sysid -1 -score sre18dev,sre18eval -backend sre04,sre05,sre06,sre08,sre10 -exclude voxceleb1,voxceleb2,noise

# No Voxceleb and fisher datasets
python train_xvec.py mfcc_musan_rirs -exclude voxceleb1,voxceleb2,fisher
python make_score.py mfcc_musan_rirs -sys xvec -sysid -1 -score sre18dev,sre18eval -backend sre04,sre05,sre06,sre08,sre10 -exclude voxceleb1,voxceleb2,fisher

