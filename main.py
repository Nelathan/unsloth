from importlib import import_module
from pathlib import Path
from rich.console import Console
from rich.table import Table
from questionary import Prompt

def main():
    #go into ./recipes and list all the recipes
    #ask user to select a recipe
    #run the selected recipe

    console = Console()
    recipes_dir = Path(__file__).parent / "recipes"
    recipes = [f.stem for f in recipes_dir.glob("*.py")]
    table = Table(title="Recipes")
    table.add_column("Index", style="cyan")
    table.add_column("Recipe", style="magenta")
    for i, recipe in enumerate(recipes):
        table.add_row(str(i), recipe)
    console.print(table)
    recipe_index = Prompt.ask("Select a recipe", choices=[str(i) for i in range(len(recipes))])
    recipe = recipes[int(recipe_index)]
    module = import_module(f"recipes.{recipe}")
    module.main()

if __name__ == "__main__":
    main()
