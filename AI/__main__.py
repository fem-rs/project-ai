from milc import cli

@cli.entrypoint("Entrypoint")
def main(cli):
    cli.log.info('No subcommand specified!')
    cli.print_usage()

@cli.argument('-m', '--model', help='File name for model to be saved as', default='alz_model.keras')
@cli.subcommand('Train AI.')
def train(cli):
    cli.echo(cli.config.train.model)


if __name__ == '__main__':
    cli()
