def test_expr_tracker():
    import expr_tracker as et

    et.init(
        project="test_project",
        name="test_run",
        dir="./tests/logs",
        config={"param": 42},
        print_to_screen=True,
        backends=["jsonl", "wandb"],
    )
    for i in range(5):
        et.log({"metric": i * 10}, step=i)
    et.finish()
