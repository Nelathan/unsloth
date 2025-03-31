from importlib import import_module
from pathlib import Path
from rich.console import Console
from rich.table import Table
import questionary

def main():
    console = Console()
    recipes_dir = Path(__file__).parent / "recipes"

    if not recipes_dir.exists():
        console.print("[bold red]Error: 'recipes' directory not found![/bold red]")
        return

    recipes = [f.stem for f in recipes_dir.glob("*.py") if f.name != "__init__.py"]

    if not recipes:
        console.print("[bold yellow]No recipes found![/bold yellow]")
        return

    table = Table(title="Recipes")
    table.add_column("Index", style="cyan")
    table.add_column("Recipe", style="magenta")

    for i, recipe in enumerate(recipes):
        table.add_row(str(i), recipe)

    console.print(table)

    recipe_index = questionary.select(
        "Select a recipe:",
        choices=[f"{i}: {recipes[i]}" for i in range(len(recipes))]
    ).ask()

    if recipe_index is None:
        console.print("[bold yellow]No selection made. Exiting.[/bold yellow]")
        return

    recipe = recipes[int(recipe_index.split(":")[0])]
    module = import_module(f"recipes.{recipe}")
    
    if not hasattr(module, "main"):
        console.print(f"[bold red]Error: The recipe '{recipe}' has no main() function![/bold red]")
        return

    module.main()

if __name__ == "__main__":
    main()