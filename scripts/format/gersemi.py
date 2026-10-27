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
        tool = values[0].strip('"')
        lowering = tool.rsplit("/", 1)[-1] in {"buddy-opt", "mlir-opt"} or (
            tool.startswith("${DESIGN_TARGET_OPT_") and tool.endswith("}")
        )
        packing = values[0].strip('"').rsplit("/", 1)[-1] == "rax-pack"
        next_tool = False
        for value in values[1:]:
            if value == "|":
                lines[-1] += " |"
                next_tool = True
            elif next_tool:
                lines.append(tool_indent + value)
                tool = value.strip('"')
                lowering = tool.rsplit("/", 1)[-1] in {
                    "buddy-opt",
                    "mlir-opt",
                } or (tool.startswith("${DESIGN_TARGET_OPT_") and tool.endswith("}"))
                next_tool = False
            elif (lowering or packing) and value.startswith("-"):
                lines.append((tool_indent if packing else pass_indent) + value)
            elif not lowering and value.startswith("--"):
                lines.append(pass_indent + value)
            else:
                lines[-1] += " " + value
        return "\n".join(lines)


class TargetPropertiesCommand(
    create_standard_dumper(
        preprocess_definitions(builtin_commands)["set_target_properties"]
    )
):
    def _inlining_condition(self, arguments):
        return False

    def _format_non_option(self, tree):
        keyword, *values = tree.children
        if get_value(keyword, None) != "PROPERTIES":
            return super()._format_non_option(tree)
        begin = self.visit(keyword)
        with self.not_indented():
            properties = self._format_keyword_with_pairs(values)
        return begin + " " + properties.replace("\n", "\n" + " " * (len(begin) + 1))

    def _format_command_with_long_name(self, begin, arguments, end):
        return self.format_command_with_short_name(begin, arguments, end)


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
        if len(arguments.children) <= 2:
            return super().format_command(tree)
        with self.not_indented():
            values = [self.visit(argument) for argument in arguments.children]
        prefix = self._indent(self.format_command_name(identifier) + "(")
        prefix += values[0] + "\n"
        with self.indented():
            keyword = self._indent(values[1] + " ")
        return (
            prefix
            + keyword
            + ("\n" + " " * len(keyword)).join(values[2:])
            + "\n"
            + self._indent(")")
        )


command_definitions = {
    name: {"__impl": CompactCommand}
    for name in (
        "file",
        "add_library",
        "add_executable",
    )
}
command_definitions["add_custom_command"] = {"__impl": BuddyCustomCommand}
command_definitions["set_target_properties"] = {"__impl": TargetPropertiesCommand}

command_definitions["target_include_directories"] = {"__impl": DependencyCommand}
command_definitions["target_link_directories"] = {"__impl": DependencyCommand}
command_definitions["target_link_libraries"] = {"__impl": DependencyCommand}
command_definitions["target_compile_definitions"] = {"__impl": DependencyCommand}
