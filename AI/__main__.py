from AI.src.predict import predict_ai
from milc import cli

from AI.src.train import train_ai

@cli.entrypoint("Entrypoint")
def main(cli):
    cli.log.info('No subcommand specified!')
    cli.print_usage()

@cli.subcommand('Train AI and save it to a file.')
def train(cli):
    train_ai()
    
@cli.subcommand('Train AI and save it to a file.')
def predict(cli):
    predict_ai()

if __name__ == '__main__':
    cli()
