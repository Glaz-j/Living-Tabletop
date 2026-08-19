from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import hashlib
import re
from typing import Iterable

from ..models import Effect, Entity, EntityType, Fact, WorldState
from .contracts import ItemChangeProposal


class ItemChangeValidationError(ValueError):
    pass


@dataclass(slots=True)
class ItemChangeMaterialization:
    effects: list[Effect] = field(default_factory=list)
    accepted: list[ItemChangeProposal] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)


class ItemChangeValidator:
    """Turns item prose proposals into bounded, auditable kernel effects."""

    _ACQUIRE_CLAIMS = re.compile(
        r"(?:递|交|塞|送|还|给)给(?:了)?你|"
        r"(?:递|推|放)到你(?:的)?(?:面前|手边|手里)|放在你面前|"
        r"你(?:接过|收下|拿起|捡起|拾起|收起|取得|获得|买下|购买|握住)|"
        r"(?:收入|塞进|装进|放进)(?:了)?(?:随身|贴身|口袋|背包)|"
        r"(?:归|属于)你|you (?:receive|take|pick up|buy|accept)",
        re.IGNORECASE,
    )
    _RELINQUISH_CLAIMS = re.compile(
        r"你(?:交出|交还|递还|丢下|放下|扔掉|卖掉|送出|用掉|吃下|喝下|消耗)|"
        r"从(?:背包|口袋|身上).{0,30}(?:递给|交给|扔掉|放下)|"
        r"you (?:drop|give away|return|consume)",
        re.IGNORECASE,
    )
    _NEGATION = re.compile(r"(?:没|没有|并未|未能|不肯|拒绝|不能|无法|别|不要)$")

    @staticmethod
    def _normalized(value: object) -> str:
        return re.sub(r"[\s，。！？、；：,.!?;:'\"“”‘’（）()·-]+", "", str(value)).lower()

    @classmethod
    def _name_is_visible(cls, item_name: str, visible_text: str) -> bool:
        normalized_name = cls._normalized(item_name)
        if normalized_name in visible_text:
            return True
        # Natural prose often shortens “诺特的私人名片” to “私人名片”.  Require
        # a meaningful contiguous overlap while still rejecting unrelated hidden
        # metadata attached to the response.
        overlap = SequenceMatcher(None, normalized_name, visible_text).find_longest_match()
        threshold = min(4, max(2, len(normalized_name) // 2))
        return overlap.size >= threshold

    @classmethod
    def _positive_claims(cls, beats: Iterable[str]) -> list[str]:
        text = "\n".join(beats)
        claims: list[str] = []
        for pattern in (cls._ACQUIRE_CLAIMS, cls._RELINQUISH_CLAIMS):
            for match in pattern.finditer(text):
                short_prefix = text[max(0, match.start() - 8) : match.start()]
                semantic_prefix = text[max(0, match.start() - 100) : match.start()]
                if cls._NEGATION.search(short_prefix):
                    continue
                # A narrator may echo the player's question or refer to a prior
                # transfer.  Those words are context, not a new world mutation.
                if re.search(
                    r"(?:关于|问的是|提到|复述|关键词|指出的)[^。！？\n]{0,96}$",
                    semantic_prefix,
                ):
                    continue
                if re.search(
                    r"(?:刚才|此前|之前|当时|先前|已经)[^。！？\n]{0,32}$",
                    semantic_prefix,
                ):
                    continue
                claims.append(match.group(0))
        return claims

    @staticmethod
    def _accessible_item(
        state: WorldState,
        item_id: str,
        *,
        allowed_entity_ids: set[str],
        counterparty_id: str | None,
    ) -> Entity | None:
        entity = state.entities.get(item_id)
        if entity is None or entity.type != EntityType.ITEM or not entity.active:
            return None
        current_location = state.entities[state.player.entity_id].location
        if (
            item_id in allowed_entity_ids
            or item_id in state.player.inventory
            or entity.location in {current_location, state.player.entity_id, counterparty_id}
        ):
            return entity
        return None

    def _find_existing_by_name(
        self,
        state: WorldState,
        name: str,
        *,
        allowed_entity_ids: set[str],
        counterparty_id: str | None,
    ) -> Entity | None:
        normalized_name = self._normalized(name)
        matches = [
            entity
            for entity in state.entities.values()
            if entity.type == EntityType.ITEM
            and self._normalized(entity.name) == normalized_name
            and self._accessible_item(
                state,
                entity.id,
                allowed_entity_ids=allowed_entity_ids,
                counterparty_id=counterparty_id,
            )
            is not None
        ]
        # Reusing an already carried or dynamic item prevents repeated dialogue
        # from cloning the same object.  Ambiguous authored items require an id.
        matches.sort(
            key=lambda entity: (
                entity.id not in state.player.inventory,
                "dynamic" not in entity.tags,
                entity.id,
            )
        )
        return matches[0] if matches else None

    @staticmethod
    def _counterparty_is_accessible(
        state: WorldState,
        counterparty_id: str | None,
        allowed_entity_ids: set[str],
    ) -> bool:
        if counterparty_id is None:
            return True
        entity = state.entities.get(counterparty_id)
        return bool(
            entity is not None
            and entity.active
            and counterparty_id in allowed_entity_ids
        )

    def _new_item(
        self,
        state: WorldState,
        proposal: ItemChangeProposal,
    ) -> Entity:
        digest = hashlib.sha1(
            (
                f"{state.session_id}|{state.version}|{self._normalized(proposal.item_name)}|"
                f"{proposal.counterparty_entity_id or ''}|{proposal.origin}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        attributes = {
            "item_kind": proposal.item_kind,
            "origin": proposal.origin,
            "description": proposal.description,
        }
        if proposal.counterparty_entity_id:
            attributes["source_entity_id"] = proposal.counterparty_entity_id
        return Entity(
            id=f"item_dynamic_{digest}",
            type=EntityType.ITEM,
            name=proposal.item_name,
            location=state.entities[state.player.entity_id].location,
            attributes=attributes,
            tags={"generated_item", proposal.item_kind, f"origin:{proposal.origin}"},
            active=True,
        )

    def _possession_fact(
        self,
        state: WorldState,
        proposal: ItemChangeProposal,
        item: Entity,
    ) -> Fact:
        counterparty = state.entities.get(proposal.counterparty_entity_id or "")
        counterparty_name = counterparty.name if counterparty else None
        if proposal.operation == "acquire":
            value = (
                f"{item.name}由{counterparty_name}交给{state.player.name}，现由{state.player.name}随身携带"
                if counterparty_name
                else f"{state.player.name}取得了{item.name}，现由其随身携带"
            )
        else:
            value = (
                f"{state.player.name}将{item.name}交给了{counterparty_name}"
                if counterparty_name
                else f"{state.player.name}不再随身携带{item.name}"
            )
        digest = hashlib.sha1(
            f"{state.session_id}|{state.version}|{proposal.operation}|{item.id}|{value}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        source_id = proposal.counterparty_entity_id or state.player.entity_id
        return Fact(
            id=f"f_item_change_{digest}",
            subject=item.id,
            predicate="possession_change",
            value=value,
            visibility="PLAYER",
            created_at=state.world_time,
            source=f"dialogue:world_change:{source_id}",
            immutable=False,
            canon="soft_canon",
        )

    def materialize(
        self,
        state: WorldState,
        proposals: Iterable[ItemChangeProposal],
        *,
        performance: list[str],
        allowed_entity_ids: set[str],
    ) -> ItemChangeMaterialization:
        proposals = list(proposals)
        visible_text = self._normalized("\n".join(performance))
        explicit_claims = self._positive_claims(performance)
        if explicit_claims and not proposals:
            raise ItemChangeValidationError(
                "performance describes an item transfer without a proposed_item_changes entry"
            )

        result = ItemChangeMaterialization()
        seen: set[tuple[str, str]] = set()
        for proposal in proposals:
            if not self._name_is_visible(proposal.item_name, visible_text):
                # Metadata that is not visible in the performance has no authority.
                continue
            if not self._counterparty_is_accessible(
                state, proposal.counterparty_entity_id, allowed_entity_ids
            ):
                raise ItemChangeValidationError(
                    f"item counterparty is not accessible: {proposal.counterparty_entity_id}"
                )

            if proposal.operation == "acquire":
                item = None
                if proposal.item_entity_id is not None:
                    item = self._accessible_item(
                        state,
                        proposal.item_entity_id,
                        allowed_entity_ids=allowed_entity_ids,
                        counterparty_id=proposal.counterparty_entity_id,
                    )
                    if item is None:
                        raise ItemChangeValidationError(
                            f"item is not accessible: {proposal.item_entity_id}"
                        )
                else:
                    item = self._find_existing_by_name(
                        state,
                        proposal.item_name,
                        allowed_entity_ids=allowed_entity_ids,
                        counterparty_id=proposal.counterparty_entity_id,
                    )
                create_item = item is None
                if item is None:
                    item = self._new_item(state, proposal)
                key = (proposal.operation, item.id)
                if key in seen:
                    continue
                seen.add(key)
                result.accepted.append(proposal)
                result.item_ids.append(item.id)
                if item.id in state.player.inventory:
                    continue
                if create_item:
                    result.effects.append(
                        Effect(op="create_entity", params={"entity": item.model_dump(mode="json")})
                    )
                result.effects.append(Effect(op="add_inventory", params={"item_id": item.id}))
            else:
                item = self._accessible_item(
                    state,
                    str(proposal.item_entity_id),
                    allowed_entity_ids=allowed_entity_ids,
                    counterparty_id=proposal.counterparty_entity_id,
                )
                if item is None or item.id not in state.player.inventory:
                    raise ItemChangeValidationError(
                        f"player cannot relinquish uncarried item: {proposal.item_entity_id}"
                    )
                key = (proposal.operation, item.id)
                if key in seen:
                    continue
                seen.add(key)
                result.accepted.append(proposal)
                result.item_ids.append(item.id)
                destination = (
                    proposal.counterparty_entity_id
                    or state.entities[state.player.entity_id].location
                )
                result.effects.append(
                    Effect(
                        op="remove_inventory",
                        params={"item_id": item.id, "destination": destination},
                    )
                )
                if proposal.origin == "consume":
                    result.effects.append(
                        Effect(
                            op="set_entity_active",
                            params={"entity_id": item.id, "active": False},
                        )
                    )

            fact = self._possession_fact(state, proposal, item)
            result.effects.extend(
                [
                    Effect(op="create_fact", params={"fact": fact.model_dump(mode="json")}),
                    Effect(op="reveal_fact", params={"fact_id": fact.id}),
                ]
            )

        if explicit_claims and not result.accepted:
            raise ItemChangeValidationError(
                "performance contains an item transfer that could not be materialized"
            )
        return result
