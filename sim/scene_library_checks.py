"""Registered scene construction stays scoped to its service instance."""
import unittest

import sim.service as service
from .scene_library import LibraryEnv, service_factory, with_initial_selection


class SceneLibraryCompositionChecks(unittest.TestCase):
    def test_library_environment_is_injected_without_replacing_service_global(self):
        class BaseService:
            def __init__(self,**kwargs): self.dependencies=kwargs

        original=service.OpenArmEnv
        instance=service_factory(BaseService)()
        self.assertIs(instance.dependencies['environment_factory'],LibraryEnv)
        self.assertIs(service.OpenArmEnv,original)

    def test_initial_selection_is_bound_and_idempotent_per_service_class(self):
        class BaseService:
            def __init__(self,**kwargs): self.dependencies=kwargs

        LibraryService=service_factory(BaseService)
        Selected=with_initial_selection(LibraryService,['block','can'])
        self.assertIs(with_initial_selection(Selected,['can','block']),Selected)
        instance=Selected()
        self.assertEqual(instance.dependencies['initial_scene_selection'],('can','block'))
        self.assertIs(instance.dependencies['environment_factory'],LibraryEnv)

    def test_conflicting_or_unsupported_scene_factory_is_rejected(self):
        class BaseService:
            def __init__(self,**kwargs): self.dependencies=kwargs
        class OtherEnv: pass

        Selected=with_initial_selection(service_factory(BaseService),['can','block'])
        with self.assertRaisesRegex(ValueError,'already belongs'):
            Selected(initial_scene_selection=('bottle',))
        with self.assertRaisesRegex(ValueError,'requires its registered LibraryEnv'):
            service_factory(BaseService)(environment_factory=OtherEnv)


if __name__=='__main__':unittest.main()
