from odin.fuel import MNIST
from odin.bay.vi import BetaGammaVAE
from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class Arguments:
  zdim: int = 10
  beta: int = 1
  gamma: int = 1


  def parse(self):
    args = ArgumentParser()
    args.add_argument('-zdim, ')

def main(args: Arguments):
  pass
