import click
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs

@click.group()
def cli():
    """Pipeline command line interface for processing PDF reports and questions."""
    pass

@cli.command()
def download_models():
    """[已废弃] 原 Docling 模型下载已移除，当前流程使用 pdf_mineru 处理 PDF。"""
    click.echo("Docling 流程已移除，请使用 pdf_mineru 相关流程处理 PDF 报告。")

@cli.command()
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker')
@click.option('--max-workers', default=10, help='Number of parallel worker processes')
def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with optional parallel processing."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"Parsing PDFs (parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers})")
    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

@cli.command()
@click.option('--max-workers', default=10, help='[已废弃] 表格序列化流程已移除')
def serialize_tables(max_workers):
    """[已废弃] 表格序列化流程已移除。"""
    click.echo("表格序列化流程已移除，该命令不再执行任何操作。")

@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='Configuration preset to use')
def process_reports(config):
    """Process parsed reports through the pipeline stages."""
    root_path = Path.cwd()
    run_config = preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing parsed reports (config={config})...")
    pipeline.process_parsed_reports()

@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking']), default='base', help='Configuration preset to use')
def process_questions(config):
    """Process questions using the pipeline."""
    root_path = Path.cwd()
    run_config = configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing questions (config={config})...")
    pipeline.process_questions()

if __name__ == '__main__':
    cli()