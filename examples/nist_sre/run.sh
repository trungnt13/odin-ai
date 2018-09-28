# For training: fisher, mx6, sre04, sre05, sre06, sre08, sre10, swb, voxceleb1, voxceleb2
# For noise   : musan, rirs

# Test training on voxceleb1
python train_xvec.py mspec_musan_rirs -exclude fisher,mx6,sre04,sre05,sre06,sre08,sre10,swb,voxceleb2
python make_score.py mspec_musan_rirs -sys xvec -sysid -1 -score voxceleb1,sre18dev -backend voxceleb1 -exclude fisher,mx6,sre04,sre05,sre06,sre08,sre10,swb,voxceleb2

# No Voxceleb datasets
python train_xvec.py mspec_musan_rirs -exclude voxceleb1,voxceleb2
python make_score.py mspec_musan_rirs -sys xvec -sysid -1 -score sre18dev,sre18eval -backend sre04,sre05,sre06,sre08,sre10 -exclude voxceleb1,voxceleb2

# same but without noise
python train_xvec.py mspec_musan_rirs -exclude voxceleb1,voxceleb2,noise
python make_score.py mspec_musan_rirs -sys xvec -sysid -1 -score sre18dev,sre18eval -backend sre04,sre05,sre06,sre08,sre10 -exclude voxceleb1,voxceleb2,noise

# No Voxceleb and fisher datasets
python train_xvec.py mspec_musan_rirs -exclude voxceleb1,voxceleb2,fisher
python make_score.py mspec_musan_rirs -sys xvec -sysid -1 -score sre18dev,sre18eval -backend sre04,sre05,sre06,sre08,sre10 -exclude voxceleb1,voxceleb2,fisher

