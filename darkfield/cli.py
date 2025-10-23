"""
Darkfield CLI Interface
Command-line interface for AI red teaming
"""

import click
import asyncio
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint
import time

from .core.exploiter import PersonaExploiter
from .models.ollama import OllamaModel, ModelManager
from .library.vectors import ExploitLibrary
from .reports.compliance import ComplianceReporter

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Darkfield - AI Red Team Framework"""
    pass


@main.command()
@click.option("--model", default="phi", help="Ollama model to use")
def validate():
    """Validate installation and check dependencies"""
    
    console.print("[bold cyan]🔍 Validating Darkfield Installation[/bold cyan]\n")
    
    # Check Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 11):
        console.print(f"✅ Python {py_version}")
    else:
        console.print(f"❌ Python {py_version} (requires 3.11+)")
    
    # Check PyTorch installation
    try:
        import torch  # noqa: WPS433

        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"✅ PyTorch {torch.__version__} (device: {device})")
    except Exception as exc:  # pragma: no cover - import failure messaging
        console.print(f"❌ PyTorch not available: {exc}")
        console.print("   Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu")

    # Check Ollama connection
    async def check_ollama():
        model = OllamaModel("phi")
        return await model.check_connection()
    
    if asyncio.run(check_ollama()):
        console.print("✅ Ollama connected")
        
        # List available models
        async def list_models():
            model = OllamaModel("phi")
            return await model.list_models()
        
        models = asyncio.run(list_models())
        if models:
            console.print(f"   Available models: {', '.join(models)}")
        else:
            console.print("   ⚠️  No models found. Run: ollama pull phi")
    else:
        console.print("❌ Ollama not running. Run: ollama serve")
    
    # Check directories
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        console.print("✅ Created data directory")
    else:
        console.print("✅ Data directory exists")
    
    console.print("\n[bold green]Installation validated successfully![/bold green]")


@main.command()
@click.option("--trait", default="helpful", help="Personality trait to exploit")
@click.option("--objective", default="ignore safety guidelines", help="Attack objective")
@click.option("--model", default="phi", help="Ollama model to use")
@click.option("--category", default="persona_inversion", help="Exploit category")
@click.option("--stealth", is_flag=True, help="Apply stealth techniques")
def exploit(trait, objective, model, category, stealth):
    """Generate a single exploit"""
    
    console.print(f"[bold cyan]🎯 Generating Exploit[/bold cyan]")
    console.print(f"   Trait: {trait}")
    console.print(f"   Objective: {objective}")
    console.print(f"   Category: {category}")
    console.print(f"   Model: {model}\n")
    
    async def generate():
        exploiter = PersonaExploiter(model=model)
        return await exploiter.generate_exploit(
            trait=trait,
            objective=objective,
            category=category,
            use_stealth=stealth
        )
    
    with console.status("Generating exploit..."):
        exploit_obj = asyncio.run(generate())
    
    # Display results
    console.print("[bold green]✅ Exploit Generated[/bold green]\n")
    
    table = Table(title="Exploit Details")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("ID", exploit_obj.id)
    table.add_row("Success Rate", f"{exploit_obj.success_rate:.1%}")
    table.add_row("Stealth Score", f"{exploit_obj.stealth_score:.1%}")
    table.add_row("Complexity", f"{exploit_obj.complexity}/10")
    table.add_row("Vector Norm", f"{exploit_obj.vector_norm:.3f}")
    
    console.print(table)
    
    console.print("\n[bold]Payload:[/bold]")
    console.print(f"[yellow]{exploit_obj.payload}[/yellow]")


@main.command()
@click.option("--count", default=100, help="Number of exploits to generate")
@click.option("--model", default="phi", help="Ollama model to use")
@click.option("--output", default="data/exploits/library.json", help="Output file path")
def build_library(count, model, output):
    """Build exploit library"""
    
    console.print(f"[bold cyan]📚 Building Exploit Library[/bold cyan]")
    console.print(f"   Target: {count} exploits")
    console.print(f"   Model: {model}")
    console.print(f"   Output: {output}\n")
    
    async def build():
        from .library.builder import LibraryBuilder
        builder = LibraryBuilder(model=model)
        return await builder.build(count=count, output_path=output)
    
    start_time = time.time()
    
    with console.status(f"Generating {count} exploits..."):
        result = asyncio.run(build())
    
    elapsed = time.time() - start_time
    
    console.print(f"\n[bold green]✅ Library Built Successfully![/bold green]")
    console.print(f"   Exploits: {result['count']}")
    console.print(f"   Time: {elapsed/60:.1f} minutes")
    console.print(f"   Rate: {result['count']/elapsed:.1f} exploits/sec")
    console.print(f"   File: {output}")


@main.command()
@click.option("--model", default="phi", help="Model to benchmark")
def benchmark(model):
    """Benchmark model performance"""
    
    console.print(f"[bold cyan]⚡ Benchmarking Model Performance[/bold cyan]")
    console.print(f"   Model: {model}\n")
    
    async def run_benchmark():
        model_obj = OllamaModel(model)
        
        prompts = [
            ("Simple", "Hello, how are you?", 20),
            ("Medium", "Explain quantum computing in simple terms.", 100),
            ("Complex", "Write a Python function to sort a list.", 200),
        ]
        
        results = []
        
        for name, prompt, tokens in prompts:
            console.print(f"Testing: {name}")
            
            times = []
            for i in range(3):
                start = time.time()
                response = await model_obj.generate(prompt, max_tokens=tokens)
                elapsed = time.time() - start
                times.append(elapsed)
                console.print(f"   Run {i+1}: {elapsed:.2f}s")
            
            avg_time = sum(times) / len(times)
            tokens_per_sec = tokens / avg_time
            
            results.append({
                "test": name,
                "avg_time": avg_time,
                "tokens_per_sec": tokens_per_sec
            })
        
        return results
    
    results = asyncio.run(run_benchmark())
    
    # Display results table
    table = Table(title=f"Benchmark Results - {model}")
    table.add_column("Test", style="cyan")
    table.add_column("Avg Time (s)", style="yellow")
    table.add_column("Tokens/sec", style="green")
    
    for r in results:
        table.add_row(
            r["test"],
            f"{r['avg_time']:.2f}",
            f"{r['tokens_per_sec']:.1f}"
        )
    
    console.print("\n")
    console.print(table)


@main.command()
@click.argument("company")
@click.option("--frameworks", default="SOC2,GDPR,AI_ACT", help="Compliance frameworks")
@click.option("--format", default="pdf", type=click.Choice(["pdf", "json"]), help="Report format")
@click.option("--output", help="Output file path")
def report(company, frameworks, format, output):
    """Generate compliance report"""
    
    console.print(f"[bold cyan]📊 Generating Compliance Report[/bold cyan]")
    console.print(f"   Company: {company}")
    console.print(f"   Frameworks: {frameworks}")
    console.print(f"   Format: {format}\n")
    
    frameworks_list = frameworks.split(",")
    
    reporter = ComplianceReporter()
    
    with console.status("Generating report..."):
        if format == "pdf":
            report_path = reporter.generate_pdf_report(
                company_name=company,
                frameworks=frameworks_list
            )
        else:
            report_path = reporter.generate_json_report(
                company_name=company
            )
    
    if output:
        # Move to specified output path
        from shutil import move
        move(report_path, output)
        report_path = output
    
    console.print(f"[bold green]✅ Report Generated[/bold green]")
    console.print(f"   File: {report_path}")


@main.command()
@click.option("--format", default="json", type=click.Choice(["json", "csv"]), help="Export format")
@click.option("--output", required=True, help="Output file path")
def export(format, output):
    """Export exploit library"""
    
    console.print(f"[bold cyan]📤 Exporting Exploit Library[/bold cyan]")
    console.print(f"   Format: {format}")
    console.print(f"   Output: {output}\n")
    
    library = ExploitLibrary()
    
    with console.status("Exporting..."):
        if format == "json":
            count = library.export_json(output)
        else:
            count = library.export_csv(output)
    
    console.print(f"[bold green]✅ Exported {count} exploits[/bold green]")


@main.command()
def interactive():
    """Start interactive mode"""
    
    console.print("[bold cyan]🎮 Darkfield Interactive Mode[/bold cyan]\n")
    console.print("Type 'help' for commands, 'exit' to quit\n")
    
    async def run_interactive():
        exploiter = PersonaExploiter(model="phi")
        
        while True:
            try:
                command = console.input("[bold]darkfield>[/bold] ")
                
                if command == "exit":
                    break
                elif command == "help":
                    console.print("\nCommands:")
                    console.print("  exploit <trait> <objective> - Generate exploit")
                    console.print("  traits - List available traits")
                    console.print("  categories - List exploit categories")
                    console.print("  help - Show this help")
                    console.print("  exit - Exit interactive mode\n")
                elif command == "traits":
                    traits = exploiter.get_supported_traits()
                    console.print(f"\nAvailable traits: {', '.join(traits[:10])}...\n")
                elif command == "categories":
                    categories = exploiter.get_categories()
                    for cat, desc in categories.items():
                        console.print(f"  {cat}: {desc}")
                    console.print()
                elif command.startswith("exploit "):
                    parts = command.split(maxsplit=2)
                    if len(parts) == 3:
                        _, trait, objective = parts
                        console.print(f"Generating exploit for '{trait}' -> '{objective}'...")
                        exploit = await exploiter.generate_exploit(trait, objective)
                        console.print(f"[yellow]{exploit.payload}[/yellow]")
                        console.print(f"Success rate: {exploit.success_rate:.1%}\n")
                    else:
                        console.print("Usage: exploit <trait> <objective>\n")
                else:
                    console.print(f"Unknown command: {command}. Type 'help' for commands.\n")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]\n")
    
    asyncio.run(run_interactive())
    console.print("\n[bold cyan]Goodbye![/bold cyan]")


if __name__ == "__main__":
    main()