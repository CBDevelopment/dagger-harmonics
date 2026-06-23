from analysis.animation import main


def test_main_runs(capsys):
    main()
    captured = capsys.readouterr()
    assert "Dagger Harmonics" in captured.out
