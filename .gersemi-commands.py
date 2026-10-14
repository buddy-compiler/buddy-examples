from gersemi.ast_helpers import get_value
from gersemi.base_command_invocation_dumper import BaseCommandInvocationDumper
from gersemi.command_line_formatter import CommandLineFormatter
from gersemi.builtin_commands import builtin_commands, preprocess_definitions
from gersemi.specializations.standard_command_dumper import create_standard_dumper


class BuddyCustomCommand(
    create_standard_dumper(
        preprocess_definitions(builtin_commands)["add_custom_command"]
    )
):
    def _format_non_option(self, tree):
        keyword, *arguments = tree.children
        name = get_value(keyword, None)
        if name not in {"OUTPUT", "DEPENDS", "BYPRODUCTS", "COMMAND"}:
            return super()._format_non_option(tree)

        prefix = self.visit(keyword)
        with self.not_indented():
            values = [self.visit(argument) for argument in arguments]

        if name in {"OUTPUT", "DEPENDS", "BYPRODUCTS"}:
            return prefix + " " + ("\n" + " " * (len(prefix) + 1)).join(values)

        lines = [prefix + " " + values[0]]
        tool_indent = " " * (len(prefix) + 1)
        pass_indent = tool_indent + "  "
        lowering = values[0].strip('"').rsplit("/", 1)[-1] in {"buddy-opt", "mlir-opt"}
        next_tool = False
        for value in values[1:]:
            if value == "|":
                lines[-1] += " |"
                next_tool = True
            elif next_tool:
                lines.append(tool_indent + value)
                lowering = value.strip('"').rsplit("/", 1)[-1] in {
                    "buddy-opt",
                    "mlir-opt",
                }
                next_tool = False
            elif lowering and value.startswith("-"):
                lines.append(pass_indent + value)
            elif not lowering and value.startswith("--"):
                lines.append(pass_indent + value)
            else:
                lines[-1] += " " + value
        return "\n".join(lines)


class CompactCommand(CommandLineFormatter, BaseCommandInvocationDumper):
    def _inlining_condition(self, arguments):
        return True

    def _format_command_with_long_name(self, begin, arguments, end):
        self.width -= len(begin)
        with self.indented():
            formatted = self._format_command_line(arguments.children).lstrip()
        self.width += len(begin)
        return self._indent(begin) + formatted + "\n" + self._indent(end)


class DependencyCommand(CompactCommand):
    def format_command(self, tree):
        identifier, arguments = tree.children
        if len(arguments.children) <= 3:
            return super().format_command(tree)
        with self.not_indented():
            values = [self.visit(argument) for argument in arguments.children]
        prefix = self._indent(self.format_command_name(identifier) + "(")
        prefix += " ".join(values[:2]) + " "
        return (
            prefix
            + ("\n" + " " * len(prefix)).join(values[2:])
            + "\n"
            + self._indent(")")
        )


command_definitions = {
    name: {"__impl": CompactCommand}
    for name in (
        "file",
        "add_library",
        "add_executable",
        "set_target_properties",
    )
}
command_definitions["add_custom_command"] = {"__impl": BuddyCustomCommand}

command_definitions["target_include_directories"] = {"__impl": DependencyCommand}
command_definitions["target_link_libraries"] = {"__impl": DependencyCommand}
