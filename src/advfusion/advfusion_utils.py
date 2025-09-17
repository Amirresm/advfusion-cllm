def detect_adapter_type(adapter_module, adapter_name):
    """
    Detect the type of adapter module between seq_bn and compacter.
    """
    if hasattr(adapter_module.adapters[adapter_name].adapter_down[0], "W_left"):
        return "compacter"
    return "seq_bn"


def get_module_modifier(device, adapter_name, freeze=False, zero=False):

    def processor(_, module):
        if hasattr(module, "adapters") and adapter_name in module.adapters:
            adapter_type = detect_adapter_type(module, adapter_name)
            if freeze:
                if adapter_type == "compacter":
                    module.adapters[adapter_name].adapter_down[
                        0
                    ].W_left.requires_grad = False
                    module.adapters[adapter_name].adapter_down[
                        0
                    ].W_right.requires_grad = False
                    module.adapters[adapter_name].adapter_down[
                        0
                    ].b.requires_grad = False
                    module.adapters[
                        adapter_name
                    ].adapter_up.W_left.requires_grad = False
                    module.adapters[
                        adapter_name
                    ].adapter_up.W_right.requires_grad = False
                    module.adapters[adapter_name].adapter_up.b.requires_grad = (
                        False
                    )
                else:
                    module.adapters[adapter_name].adapter_down[
                        0
                    ].weight.requires_grad = False
                    module.adapters[adapter_name].adapter_down[
                        0
                    ].bias.requires_grad = False

                    module.adapters[
                        adapter_name
                    ].adapter_up.weight.requires_grad = False
                    module.adapters[
                        adapter_name
                    ].adapter_up.bias.requires_grad = False

            if zero:
                if adapter_type == "compacter":
                    module.adapters[adapter_name].adapter_down[
                        0
                    ].W_left.data.fill_(0)
                    module.adapters[adapter_name].adapter_down[
                        0
                    ].W_right.data.fill_(0)
                    module.adapters[adapter_name].adapter_down[0].b.data.fill_(
                        0
                    )
                    module.adapters[adapter_name].adapter_up.W_left.data.fill_(
                        0
                    )
                    module.adapters[adapter_name].adapter_up.W_right.data.fill_(
                        0
                    )
                    module.adapters[adapter_name].adapter_up.b.data.fill_(0)
                else:
                    module.adapters[adapter_name].adapter_down[
                        0
                    ].weight.data.fill_(0)
                    module.adapters[adapter_name].adapter_down[
                        0
                    ].bias.data.fill_(0)
                    module.adapters[adapter_name].adapter_up.weight.data.fill_(
                        0
                    )
                    module.adapters[adapter_name].adapter_up.bias.data.fill_(0)

    return processor


def reload_adapter(model, adapter_path, adapter_name, dtype):
    print(f"Reloading adapter {adapter_name} from {adapter_path}")
    model.load_adapter(
        adapter_path,
        load_as=adapter_name,
        set_active=True,
    )
    model.adapter_to(adapter_name, device=model.device, dtype=dtype)


def freeze_adapter(model, adapter_name, freeze=True):
    processor = get_module_modifier(
        model.device, adapter_name, freeze=freeze, zero=False
    )
    model.apply_to_adapter_layers(processor)


def zero_adapter(model, adapter_name):
    processor = get_module_modifier(
        model.device, adapter_name, freeze=False, zero=True
    )
    model.apply_to_adapter_layers(processor)
