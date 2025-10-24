from shutil import get_terminal_size


def heading(
	text: str,
	*,
	char: str = "─",
	width: int | None = None,
	align: str = "center",
	pad: int = 0,
	color: str | None = None,
):
	"""
	Print a clean console heading with optional width, alignment, padding, and ANSI color.
	align: 'left' | 'center' | 'right'
	color: one of ('red','green','yellow','blue','magenta','cyan','white','bright_black') or None
	"""
	cols = width or max(40, min(120, get_terminal_size((80, 24)).columns))
	line = (char * cols)[:cols]

	if align == "left":
		line_text = text.ljust(cols)
	elif align == "right":
		line_text = text.rjust(cols)
	else:
		line_text = text.center(cols)

	colors = {
		"red": 31,
		"green": 32,
		"yellow": 33,
		"blue": 34,
		"magenta": 35,
		"cyan": 36,
		"white": 37,
		"bright_black": 90,
	}

	def tint(s: str) -> str:
		return f"\033[1;{colors[color]}m{s}\033[0m" if color in colors else s

	print(line)
	for _ in range(pad):
		print()
	print(tint(line_text))
	for _ in range(pad):
		print()
	print(line)
