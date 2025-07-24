#ifndef LIFTED_WAVELETS_H_
#define LIFTED_WAVELETS_H_

#include <tuple>
#include <array>
#include <unordered_map>
#include <type_traits>
#include "lifted-common.hpp"

namespace lifted {

	template<typename T>
	class Lazy{
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple();
		const static size_t width = 1;
	};

	// Daubechies Type Wavelets
    template<typename T>
    class Daubechies1{

		constexpr static auto s1 = detail::unit_update_d<T, detail::UpdateOperation::sub>(0);
		constexpr static auto s2 = detail::update_s<T>(0, 0.5L);

        constexpr static auto sc = detail::ScaleStep<T>(1.41421356237309504880168872420969807856967187537694807317668L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, sc
		);
        const static size_t width = 2;
    };

    template<typename T>
    class Daubechies2{
    private:

		constexpr static auto s1 = detail::update_d<T>(
            0, -1.73205080756887729352744634150587236694280525381038062805581L
        );
		constexpr static auto s2 = detail::update_s<T>(
            0,
            0.433012701892219323381861585376468091735701313452595157013952L,
            -0.0669872981077806766181384146235319082642986865474048429860483L
        );
		constexpr static auto s3 = detail::unit_update_d<T, detail::UpdateOperation::add>(-1);

        constexpr static auto sc = detail::ScaleStep<T>(
            1.93185165257813657349948639945779473526780967801680910080469L
        );
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, sc
		);

        const static size_t width = 4;
    };

	template<typename T>
	class Daubechies3{
	private:

		constexpr static auto s1 = detail::update_s<T>(
            0,
			2.42549724391195830853534735792805013148173539788185404597169L
		);
		constexpr static auto s2 = detail::update_d<T>(
            -1,
			0.0793394561851575665709854468556681022464175974577020488100928L,
			-0.352387657674855469319634307041734500360168717199626110489802L
		);
		constexpr static auto s3 = detail::update_s<T>(
            1,
			-2.89534745414509893429485976731702946529529368748135552662476L,
			0.561414909153505374525726237570322812332695442156902611352901L
		);
		constexpr static auto s4 = detail::update_d<T>(
            -2,
			-0.019750529242293006004979050052766598262001873036524279141228L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			0.431879991517282793698835833676951360096586647014001193148804L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, sc
		);
		const static size_t width = 6;
	};

	template<typename T>
	class Daubechies4{
	private:

		constexpr static auto s1 = detail::update_d<T>(
            0,
			-3.10293148583032589470063811758833883204289885795495381963662L
		);

		constexpr static auto s2 = detail::update_s<T>(
            0,
			0.291953126003475318556841072165741485752300908535736046765408L,
			-0.0763000865730822647117716642303829704900132630734716898217313L
		);
		constexpr static auto s3 = detail::update_d<T>(
            -2,
			-1.66252835290918420695309216797834224617770151758624450562630L,
			5.19949157307254554931197031659870769733798631513761235614244L
		);
		constexpr static auto s4 = detail::update_s<T>(
            2,
			0.0378927481279514768107649771704322007124977302352613448672599L,
			-0.00672237263307937367389807287370832440934919551535679032055690L
		);
		constexpr static auto s5 = detail::update_d<T>(
            -3,
			0.3141064933959557065391048234478969083531725328477611313704L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			2.61311836977700528153533622961553870584070279281549965816974L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, sc
		);
		const static size_t width = 8;
	};

	template<typename T>
	class Daubechies5{
	private:

		constexpr static auto s1 = detail::update_s<T>(
            0,
			3.77151921168926894757552366178790881698024090635846510016028L
		);

		constexpr static auto s2 = detail::update_d<T>(
            -1,
			0.0698842923341383375213513979361008347245073868657407740184605L,
			-0.247729291360329682294999795251047782875275189599695961483240L
		);

		constexpr static auto s3 = detail::update_s<T>(
            1,
			-7.59757973540575405854153663362110608451687695913930970651012L,
			3.03368979135389213275426841861823490779263829963269704719700L
		);
		constexpr static auto s4 = detail::update_d<T>(
            -3,
			0.0157993238122452355950658463497927543465183810596906799882017L,
			-0.0503963526115147652552264735912933014908335872176681805787989L
		);
		constexpr static auto s5 = detail::update_s<T>(
            3,
			-1.10314632826313117312599754225554155377077047715385824307618L,
			0.172572555625945876870245572286755166281402778242781642128057L
		);
		constexpr static auto s6 = detail::update_d<T>(
            -4,
			-0.002514343828207810116980039456958207159821965712982241017850L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			0.347389040193226710460416754871479443615028875117970218575475L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, sc
		);
		const static size_t width = 10;
	};

	template<typename T>
	class Daubechies6{
	private:

		constexpr static auto s1 = detail::update_d<T>(
            0,
			-4.43446829869067456522902386901421595117929955784397002766726L
		);

		constexpr static auto s2 = detail::update_s<T>(
            0,
			0.214593450003008209156010000739498614149390366235009628657503L,
			-0.0633131925460940277584543253045170400260191584783746185619816L
		);
		constexpr static auto s3 = detail::update_d<T>(
            -2,
			-4.49311316887101738480447295831022985715670304865394507715493L,
			9.97001561279871984191261047001888284698487613742269103059835L
		);
		constexpr static auto s4 = detail::update_s<T>(
            2,
			0.0574139368483299602834412946783536920860387960959081807080617L,
			-0.0236634936624266945597991164435457276455333750966623043873005L
		);
		constexpr static auto s5 = detail::update_d<T>(
            -4,
			-0.678784346153377860800578862111733303605730559399948511853599L,
			2.35649702197482872471682359887683553808556485181375033959333L
		);
		constexpr static auto s6 = detail::update_s<T>(
            4,
			0.00718356311583346870790647002569199987967970725769484660306772L,
			-0.000991165530519446158731723517460535223234500789445981449538906L
		);
		constexpr static auto s7 = detail::update_d<T>(
            -5,
			0.0941066740419976307126245842066834084085931268118251639620L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			3.12146472110567217396698691201148558733292298702703900943590L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, s7, sc
		);
		const static size_t width = 12;
	};

	template<typename T>
	class Daubechies7{
	private:

		constexpr static auto s1 = detail::update_s<T>(
            0,
			5.09349848430361493158448369309319682065288729718818488170086L
		);

		constexpr static auto s2 = detail::update_d<T>(
            -1,
			0.0573987259809472316318163158166229520747344869480313379225195L,
			-0.189042092071992120719115069286184373462757526526706607687359L
		);

		constexpr static auto s3 = detail::update_s<T>(
            1,
			-12.2854449967343086359903552773950797906521905606178682213636L,
			5.95920876247802982415703053009457554606483451817620285866937L
		);
		constexpr static auto s4 = detail::update_d<T>(
            -3,
			0.0291354832120604111715672746297275711217338322811099054777142L,
			-0.0604278631317421373037302780741631438456816254617632977400450L
		);
		constexpr static auto s5 = detail::update_s<T>(
            3,
			-3.97071066749950430300434793946545058865144871092579097855408L,
			1.56044025996325478842482192099426071408509934413107380064471L
		);
		constexpr static auto s6 = detail::update_d<T>(
            -5,
			0.00330657330114293172083753191386779325367868962178489518206046L,
			-0.0126913773899277263576544929765396597865052704453285425832652L
		);
		constexpr static auto s7 = detail::update_s<T>(
            5,
			-0.414198450444716993956500507305024397477047469411911796987340L,
			0.0508158836433382836486473921674908683852045493076552967984957L
		);
		constexpr static auto s8 = detail::update_d<T>(
            -6,
			-0.000406214488767545513490343915060150692174820527241063790923L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			0.299010707585297416977548850152193565671577389737970684943879L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, s7, s8, sc
		);
		const static size_t width = 14;
	};

	template<typename T>
	class Daubechies8{
	private:

		constexpr static auto s1 = detail::update_d<T>(
            0,
			-5.74964161202428952531641742297874447770069863700611439780625L
		);

		constexpr static auto s2 = detail::update_s<T>(
            0,
			0.189042092071992120719115069286184373462757526526706607687359L,
			-0.0522692017709505657615046200264570878753855329566801514443159L
		);
		constexpr static auto s3 = detail::update_d<T>(
            -2,
			-7.40210683018195497281259144457822959078824042693518644132815L,
			14.5428209935956353268617650552874052350090521140251529737701L
		);
		constexpr static auto s4 = detail::update_s<T>(
            2,
			0.0609092565131526712500223989293871769934896033103158195108610L,
			-0.0324020739799994261972421468359233560093660827487021893317097L
		);
		constexpr static auto s5 = detail::update_d<T>(
            -4,
			-2.75569878011462525067005258197637909961926722730653432065204L,
			5.81871648960616924395028985550090363087448278263304640355032L
		);
		constexpr static auto s6 = detail::update_s<T>(
            4,
			0.0179741053616847069172743389156784247398807427207560577921141L,
			-0.00655821883883417186334513866087578753320413229517368432598099L
		);
		constexpr static auto s7 = detail::update_d<T>(
            -6,
			-0.247286508228382323711914584543135429166229548148776820566842L,
			1.05058183627305042173272934526705555053668924546503033598454L
		);
		constexpr static auto s8 = detail::update_s<T>(
            6,
			0.00155378941435181290328472355847245468306658119261351754857778L,
			-0.000171095706397052251935603348141260065760590013485603128919513L
		);
		constexpr static auto s9 = detail::update_d<T>(
            -7,
			0.0272403229721260106412588252370521928932024767680952595018L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			3.55216212499884228308604950160526405248328786633495753507726L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, s7, s8, s9, sc
		);
		const static size_t width = 18;
	};

	template<typename T>
	class Daubechies9{
	private:

		constexpr static auto s1 = detail::update_s<T>(
            0,
			6.40356667029530508760080808783489133464036532617193668200642L
		);

		constexpr static auto s2 = detail::update_d<T>(
            -1,
			0.0478620457576149696142136339183134192664759465250407879952019L,
			-0.152445307138113158125515923369521189954909195320469555091745L
		);

		constexpr static auto s3 = detail::update_s<T>(
            1,
			-16.7495289773888798826988564047533244498790443964285935409250L,
			8.81374755622085106660660863038249265306851411409730451179401L
		);
		constexpr static auto s4 = detail::update_d<T>(
            -3,
			0.0340347707916009652867818067005911583675728508345200913918348L,
			-0.0599208900033806032126553368487307560798208913547861697203664L
		);
		constexpr static auto s5 = detail::update_s<T>(
            3,
			-7.79465596858466751790439766158850497420595357628693657112631L,
			4.16556856168973137548378360164009504192377781136783738852484L
		);
		constexpr static auto s6 = detail::update_d<T>(
            -5,
			0.0101058807409283297084867372477325233905693061878794566228052L,
			-0.0224419008121378002481343456879309753491248147469986492112145L
		);
		constexpr static auto s7 = detail::update_s<T>(
            5,
			-2.02686517288325962820838159403507719850692398840316366775084L,
			0.680385404820989371520060130779157801338964153839808884599304L
		);
		constexpr static auto s8 = detail::update_d<T>(
            -7,
			0.000737614555567870951119489872174219864600074362045135172869561L,
			-0.00345517675942357584563854529369350464317538313241706164777921L
		);
		constexpr static auto s9 = detail::update_s<T>(
            7,
			-0.145575980707287404465639734807644548395600047373340146812474L,
			0.0145160493659163017396796133362259414298078142573013650468794L
		);
		constexpr static auto s10 = detail::update_d<T>(
            -8,
			-0.000073558753801024382174422512593349405711068832718470204911L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			0.266806475394164811023784074087480714665703180413524153545256L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, sc
		);
		const static size_t width = 18;
	};
	
	template<typename T>
	class Daubechies10{
	private:

		constexpr static auto s1 = detail::update_d<T>(
            0,
			-7.05573271641722425372287287671537054274962482939774393309597L
		);

		constexpr static auto s2 = detail::update_s<T>(
            0,
			0.138937875273882486867801529794954198889605368487685062499916L,
			-0.0440723855473913159199751979923158273798120987048451923439931L
		);
		constexpr static auto s3 = detail::update_d<T>(
            -2,
			-10.1944019324491108447120691883139389311502922162321182217969L,
			18.9141327490851728118176967423708811816959438636030989932862L
		);
		constexpr static auto s4 = detail::update_s<T>(
            2,
			0.0581400886463108620418714676575886465057572425807448620563261L,
			-0.0345692907775274607658344294544977358985093127259602337375023L
		);
		constexpr static auto s5 = detail::update_d<T>(
            -4,
			-5.70138711246923678994802131608360234141050792847673535020183L,
			9.82667061849089314269898456882699657839357508456034855942949L
		);
		constexpr static auto s6 = detail::update_s<T>(
            4,
			0.0258915667905536217737090435506775540986604850095995114020343L,
			-0.0134214601881666825322224865563089256723608595115204041422517L
		);
		constexpr static auto s7 = detail::update_d<T>(
            -6,
			-1.39809421892900296812835643251929202203521241237059776271413L,
			3.31309217077838642695523164256389983851609417600879502794343L
		);
		constexpr static auto s8 = detail::update_s<T>(
            6,
			0.00587858298639691456648277376962090120201222076185998953606526L,
			-0.00181242379712518927320452687498457116493703409472903720690277L
		);
		constexpr static auto s9 = detail::update_d<T>(
            -8,
			-0.0845508291568382642128739817221491217957002914094375371425917L,
			0.434275883081245282445872075043694638789242093593484235654802L
		);
		constexpr static auto s10 = detail::update_s<T>(
            8,
			0.000353135120384141769320697765492599497299036748377472493001024L,
			-0.0000321412109048260097331298099101743753319602611038821674298915L
		);
		constexpr static auto s11 = detail::update_d<T>(
            -9,
			0.0076957695402440469213874787289877569218407678080231515098L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			3.93366550987980032644543137065956466690284578093101997017002L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, sc
		);
		const static size_t width = 20;
	};

	// Symlets
	// Symlets 1 - 3 are equivalent to Daubechies 1-3
	template<typename T>
	class Symlet4{
	private:

		constexpr static auto s1 = detail::update_d<T>(
            0,
			-0.391146941968911101169706692515632751259058535597815675180431L
		);

		constexpr static auto s2 = detail::update_s<T>(
            0,
			0.339243991864842623328165001547133802687835755958618749535133L,
			0.124390282933868332415703927218357743404377152617636012350602L
		);
		constexpr static auto s3 = detail::update_d<T>(
            -1,
			-4.50101288771660368163498822116058860103451852885951903340095L,
			-0.899146062976323325778084899723778664102936196078891129058605L
		);
		constexpr static auto s4 = detail::update_s<T>(
            1,
			0.230468835791147983607765482621102723812272960820671135564110L,
			-0.123157724835583262979144489911800605082224238516219507704212L
		);
		constexpr static auto s5 = detail::update_d<T>(
            -3,
			2.3274080754680838707869237980224327175542304989628734191010L,
			8.1196693210678382980711861480948211209955877607809497053267
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			2.33931640934712142733326058634102740643598665279865266064989L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, sc
		);
		const static size_t width = 8;
	};

	template<typename T>
	class Symlet5{
	private:

		constexpr static auto s1 = detail::update_s<T>(
            0,
			-1.07999184552244205483185284057003478430985338717502833350360L
		);

		constexpr static auto s2 = detail::update_d<T>(
            -1,
			1.88386645464169832099399241283310744289319800043186398600428L,
			0.498523184228631455427291538605116258495173700502855949638102L
		);

		constexpr static auto s3 = detail::update_s<T>(
            1,
			-0.500758424930526233522586137212740679721868118909000306836389L,
			-0.148051697590958820892278414264354913337220985158458238806001L
		);
		constexpr static auto s4 = detail::update_d<T>(
            -2,
			6.56271739345355756476694356688833679296553096326663748461782L,
			0.930110882056510989744927083429807287462258952062979423593507L
		);
		constexpr static auto s5 = detail::update_s<T>(
            2,
			-0.390844944814085368441306409720244124359530775775508094623140L,
			1.95073953705966435482292423890787051606616010552463566213095L
		);
		constexpr static auto s6 = detail::update_d<T>(
            -4,
			-0.0679736922677244445376830393947544982654784453526605436837L,
			-0.5126260994880396964593094921307753414607863725401694686689L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			-0.220432740550017961004611438734771521657905782609878849013999L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, sc
		);
		const static size_t width = 10;
	};

	template<typename T>
	class Symlet6{
	private:

		constexpr static auto s1 = detail::update_d<T>(
            0,
			-0.226609147605409287359847341722415005776291916297294175212125L
		);

		constexpr static auto s2 = detail::update_s<T>(
            0,
			0.215540761821108706785053528728191032226978090682496870363652L,
			-1.26706860375518912813716090259116051345582714787178201216972L
		);
		constexpr static auto s3 = detail::update_d<T>(
            -2,
			-4.25515842260650138508426401221024084863255933611118932490345L,
			0.504775726388588631077609803153024868095759782519904128209010L
		);
		constexpr static auto s4 = detail::update_s<T>(
            2,
			0.233159935346847558363596753361184362455655490704426128088538L,
			0.0447459687134733788979277716400630728289769037434894682103239L
		);
		constexpr static auto s5 = detail::update_d<T>(
            -4,
			6.62445725054736204831840693766907604987941051520730090643612L,
			-18.3890008539634560837015804872457080227100900758323725429176L
		);
		constexpr static auto s6 = detail::update_s<T>(
            4,
			-0.0567684937264354490663013330970787766403734661631984412292939L,
			-0.0370294478371382528720465911911593529775544310269365363490167L
		);
		constexpr static auto s7 = detail::update_d<T>(
            -5,
			5.511934418115697573945659805975335562503382845459883526847L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			3.29915938496773781622852198069765608906408285387288254988735L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, s7, sc
		);
		const static size_t width = 12;
	};
	
	//Coiflets
	template<typename T>
	class Coiflet2{
	private:

		constexpr static auto s1 = detail::update_d<T>(
            0,
			-0.395209488620082496004159132666490894552288363415453900216654L
		);

		constexpr static auto s2 = detail::update_s<T>(
            -1,
			-0.486553126281547010786746824168711620285801508771742550569989L,
			0.341820379066459914568789621386323683754649346991058907358568L
		);
		constexpr static auto s3 = detail::update_d<T>(
            0,
			0.102356384806853842915274696854501590560191849184372502656528L,
			0.494061820549506459101851255974592709923999977431307490745365L
		);
		constexpr static auto s4 = detail::update_s<T>(
            -1,
			1.47972869896987641707870887739437188202562216985992819545233L,
			-0.130921963832076549320780392055482739738805961634075071050064L
		);
		constexpr static auto s5 = detail::update_d<T>(
            0,
			-0.0525113427816146243003828425183171152296789709847788385651133L,
			-0.428715989638527098291905096234179418584820093349915622255789L
		);
		constexpr static auto s6 = detail::update_s<T>(
            0,
			0.483146734985798497613381610484758725546915123937859414354596L,
			-0.131670388034750104759408878071458050022809357724715522617130L
		);
		constexpr static auto s7 = detail::update_d<T>(
            -1,
			0.014654934661776989040780649404570317279952910755114196564726L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			0.577316851481330848594709432505139906969764643855150600925444L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, s7, sc
		);
		const static size_t width = 12;
	};

	template<typename T>
	class Coiflet3{
	private:

		constexpr static auto s1 = detail::update_s<T>(
            0,
			0.487435382344508708148481615423261580043249253724144121712082L
		);

		constexpr static auto s2 = detail::update_d<T>(
            0,
			-0.393857498472960538058223754725611118041069760886393899892545L,
			-0.722045211962627393161764933071704304427318900055855982640200L
		);

		constexpr static auto s3 = detail::update_s<T>(
            -2,
			0.176895180417438691319466964497403743851477067155469605065098L,
			0.614901419751021552364401779795922248885211185930303742259262L
		);
		constexpr static auto s4 = detail::update_d<T>(
            1,
			-0.175894289844152074795253976858917586589498125832393775838256L,
			-0.350427076084109752529086316194163840850785234335336088341348L
		);
		constexpr static auto s5 = detail::update_s<T>(
            -2,
			-0.318293768639612837553988348884180031465155453356753276017832L,
			0.0931087791943463981011379504924907102584083637682321798362212L
		);
		constexpr static auto s6 = detail::update_d<T>(
            1,
			0.523656011729245329840796161335259913605333495389884875981612L,
			0.511737199795411510336620033738730537388717206809666528220883L
		);
		constexpr static auto s7 = detail::update_s<T>(
            -1,
			-0.323226117080883992578893018291042698804097533102781401623564L,
			0.0858805317470458541499641440302254197897327898712691980966340L
		);
		constexpr static auto s8 = detail::update_d<T>(
            -1,
			0.0915571093529647465037308891176833398483289950985347178947069L,
			-0.165107469913914818165969138904063342821128408434795338706313L
		);
		constexpr static auto s9 = detail::update_s<T>(
            1,
			-0.0480956280541597147485082656988086823533172821316601350299984L,
			0.00659599223911741763912920939610109756140739827566339330121709L
		);
		constexpr static auto s10 = detail::update_d<T>(
            -2,
			-0.01261093011089751538390292007716781560223897824851528556619L
		);

		constexpr static auto sc = detail::ScaleStep<T>(
			1.17586567789200059336384091899546177866797267169377747320359L
		);

	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, sc
		);
		const static size_t width = 18;
	};

	// Biorthogonal
	// Bior1_1 is db1
    template<typename T>
    class Bior1_3{

		constexpr static auto s1 = detail::unit_update_d<T, detail::UpdateOperation::sub>(0);
		constexpr static auto s2 = detail::update_s<T>(-1, 0.0625L, 0.5L, -0.0625L);

        constexpr static auto sc = detail::ScaleStep<T>(1.41421356237309504880168872420969807856967187537694807317668L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, sc
		);
        const static size_t width = 6;
    };

	template<typename T>
    class Bior1_5{

		constexpr static auto s1 = detail::unit_update_d<T, detail::UpdateOperation::sub>(0);
		constexpr static auto s2 = detail::update_s<T>(-1, -0.01171875L, 0.0859375L, 0.5L, -0.0859375L, 0.01171875L);

        constexpr static auto sc = detail::ScaleStep<T>(1.41421356237309504880168872420969807856967187537694807317668L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, sc
		);
        const static size_t width = 10;
    };

	template<typename T>
    class Bior2_2 {
		constexpr static auto s1 = detail::repeat_update_d<T, 2>(0, -0.5L);
		constexpr static auto s2 = detail::repeat_update_s<T, 2>(-1, 0.25L);

        constexpr static auto sc = detail::ScaleStep<T>(1.41421356237309504880168872420969807856967187537694807317668L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, sc
		);
        const static size_t width = 5;
    };

	template<typename T>
    class Bior2_4 {
		constexpr static auto s1 = detail::repeat_update_d<T, 2>(0, -0.5L);
		constexpr static auto s2 = detail::update_s<T>(-2, -0.046875L, 0.296875L, 0.296875L, -0.046875L);

        constexpr static auto sc = detail::ScaleStep<T>(1.41421356237309504880168872420969807856967187537694807317668L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, sc
		);
        const static size_t width = 9;
    };

	template<typename T>
    class Bior2_6{
		constexpr static auto s1 = detail::repeat_update_d<T, 2>(0, -0.5L);
		constexpr static auto s2 = detail::update_s<T>(-3,
			0.009765625L, -0.076171874L, 0.31640625L, 0.31640625L, -0.076171875L, 0.009765625L
		);

        constexpr static auto sc = detail::ScaleStep<T>(1.41421356237309504880168872420969807856967187537694807317668L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, sc
		);
        const static size_t width = 13;
    };

	template<typename T>
    class Bior2_8{
		constexpr static auto s1 = detail::repeat_update_d<T, 2>(0, -0.5L);
		constexpr static auto s2 = detail::update_s<T>(-4,
			-0.00213623046875L, 0.02044677734375L, -0.09539794921875L, 0.32708740234375L, 0.32708740234375L, -0.09539794921875L, 0.02044677734375L, -0.00213623046875L
		);

        constexpr static auto sc = detail::ScaleStep<T>(1.41421356237309504880168872420969807856967187537694807317668L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, sc
		);
        const static size_t width = 17;
    };

	template<typename T>
    class Bior3_1{
		constexpr static auto s1 = detail::update_s<T>(1, -0.333333333333333333333333333333333333333333333333333333333333L);
		constexpr static auto s2 = detail::update_s<T>(-1, -0.375L, 1.125L);

        constexpr static auto sc = detail::ScaleStep<T>(0.942809041582063365867792482806465385713114583584632048784453L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, sc
		);
        const static size_t width = 4;
    };

	template<typename T>
    class Bior3_3{
		constexpr static auto s1 = detail::update_s<T>(1, -0.333333333333333333333333333333333333333333333333333333333333L);
		constexpr static auto s2 = detail::update_d<T>(-1, -0.375L, -1.125L);
		constexpr static auto s3 = detail::update_s<T>(-1,
			-0.0833333333333333333333333333333333333333333333333333333333333L,
			0.444444444444444444444444444444444444444444444444444444444444L,
			0.0833333333333333333333333333333333333333333333333333333333333L
		);

        constexpr static auto sc = detail::ScaleStep<T>(2.12132034355964257320253308631454711785450781306542210976502L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, sc
		);
        const static size_t width = 8;
    };

	template<typename T>
    class Bior3_5 {
		constexpr static auto s1 = detail::update_s<T>(1, -0.333333333333333333333333333333333333333333333333333333333333L);
		constexpr static auto s2 = detail::update_d<T>(-1, -0.375L, -1.125L);
		constexpr static auto s3 = detail::update_s<T>(-2,
			0.0173611111111111111111111111111111111111111111111111111111111L,
			-0.118055555555555555555555555555555555555555555555555555555556L,
			0.444444444444444444444444444444444444444444444444444444444444L,
			0.118055555555555555555555555555555555555555555555555555555556L,
			-0.0173611111111111111111111111111111111111111111111111111111111L
		);

        constexpr static auto sc = detail::ScaleStep<T>(2.12132034355964257320253308631454711785450781306542210976502L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, sc
		);
        const static size_t width = 12;
    };

	template<typename T>
    class Bior3_7{
		constexpr static auto s1 = detail::update_s<T>(1, -0.333333333333333333333333333333333333333333333333333333333333L);
		constexpr static auto s2 = detail::update_d<T>(-1, -0.375L, -1.125L);
		constexpr static auto s3 = detail::update_s<T>(-3,
			-0.00379774305555555555555555555555555555555555555555555555555556L,
			0.0325520833333333333333333333333333333333333333333333333333333L,
			-0.137044270833333333333333333333333333333333333333333333333333L,
			0.444444444444444444444444444444444444444444444444444444444444L,
			0.137044270833333333333333333333333333333333333333333333333333L,
			-0.0325520833333333333333333333333333333333333333333333333333333L,
			0.00379774305555555555555555555555555555555555555555555555555556L
		);

		constexpr static auto sc = detail::ScaleStep<T>(2.12132034355964257320253308631454711785450781306542210976502L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, sc
		);
        const static size_t width = 16;
    };

	template<typename T>
    class Bior3_9{
		constexpr static auto s1 = detail::update_s<T>(1, -0.333333333333333333333333333333333333333333333333333333333333L);
		constexpr static auto s2 = detail::update_d<T>(-1, -0.375L, -1.125L);
		constexpr static auto s3 = detail::update_s<T>(-3,
			0.0008544921875L,
			-0.00892469618055555555555555555555555555555555555555555555555556L,
			0.0445149739583333333333333333333333333333333333333333333333333L,
			-0.149007161458333333333333333333333333333333333333333333333333L,
			0.444444444444444444444444444444444444444444444444444444444444L,
			0.149007161458333333333333333333333333333333333333333333333333L,
			-0.0445149739583333333333333333333333333333333333333333333333333L,
			0.00892469618055555555555555555555555555555555555555555555555556L,
			-0.0008544921875L
		);

		constexpr static auto sc = detail::ScaleStep<T>(2.12132034355964257320253308631454711785450781306542210976502L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, sc
		);
        const static size_t width = 20;
    };

	template<typename T>
    class Bior4_2{
		constexpr static auto s1 = detail::repeat_update_s<T, 2>(-1, -0.25L);
		constexpr static auto s2 = detail::repeat_update_d<T, 2>(0, -1.0L);
		constexpr static auto s3 = detail::repeat_update_s<T, 2>(-1, 0.1875L);

		constexpr static auto sc = detail::ScaleStep<T>(2.82842712474619009760337744841939615713934375075389614635336L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, sc
		);
        const static size_t width = 7;
    };

	template<typename T>
    class Bior4_4{
		constexpr static auto s1 = detail::repeat_update_s<T, 2>(-1, -0.25L);
		constexpr static auto s2 = detail::repeat_update_d<T, 2>(0, -1.0L);
		constexpr static auto s3 = detail::update_s<T>(-2, -0.0390625L, 0.226562L, 0.226562L, -0.0390625L);

		constexpr static auto sc = detail::ScaleStep<T>(2.82842712474619009760337744841939615713934375075389614635336L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, sc
		);
        const static size_t width = 11;
    };

	template<typename T>
    class Bior4_6{
		constexpr static auto s1 = detail::repeat_update_s<T, 2>(-1, -0.25L);
		constexpr static auto s2 = detail::repeat_update_d<T, 2>(0, -1.0L);
		constexpr static auto s3 = detail::update_s<T>(-3, 
			0.008544921875L, -0.064697265625L, 0.24365234375L, 0.24365234375L, -0.064697265625L, 0.008544921875L
		);

		constexpr static auto sc = detail::ScaleStep<T>(2.82842712474619009760337744841939615713934375075389614635336L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, sc
		);
        const static size_t width = 15;
    };

	template<typename T>
    class Bior5_5{
		constexpr static auto s1 = detail::update_d<T>(0, -0.2L);
		constexpr static auto s2 = detail::update_s<T>(0,
			-0.208333333333333333333333333333333333333333333333333333333333L,
			0.625L
		);
		constexpr static auto s3 = detail::update_d<T>(-1, -0.9L, -1.5L);
		constexpr static auto s4 = detail::update_s<T>(-2, 
			0.0151909722222222222222222222222222222222222222222222222222222L,
			-0.0998263888888888888888888888888888888888888888888888888888889L,
			0.333333333333333333333333333333333333333333333333333333333333L,
			0.0998263888888888888888888888888888888888888888888888888888889L,
			-0.0151909722222222222222222222222222222222222222222222222222222L
		);

		constexpr static auto sc = detail::ScaleStep<T>(4.24264068711928514640506617262909423570901562613084421953004L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, sc
		);
        const static size_t width = 14;
    };

	template<typename T>
    class Bior6_8{
		constexpr static auto s1 = detail::repeat_update_d<T, 2>(0, -0.166666666666666666666666666666666666666666666666666666666667L);
		constexpr static auto s2 = detail::repeat_update_s<T, 2>(-1, -0.5625L);
		constexpr static auto s3 = detail::repeat_update_d<T, 2>(0, -1.33333333333333333333333333333333333333333333333333333333333L);
		constexpr static auto s4 = detail::update_s<T>(-4,
			-0.00176239013671875L,
			0.01650238037109375L,
			-0.07311248779296875L,
			0.21462249755859375L,
			0.21462249755859375L,
			-0.07311248779296875L,
			0.01650238037109375L,
			-0.00176239013671875L
		);

		constexpr static auto sc = detail::ScaleStep<T>(5.65685424949238019520675489683879231427868750150779229270672L);
    public:
        using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, sc
		);
        const static size_t width = 21;
    };

	// Traditional CDF wavelets
	template<typename T>
	class CDF5_3{
		// This is very close to the Bior2_2
		constexpr static auto s1 = detail::repeat_update_s<T, 2>(-1, 0.5L);
		constexpr static auto s2 = detail::repeat_update_d<T, 2>(0, -0.25L);
		constexpr static auto sc = detail::ScaleStep<T>(0.707106781186547524400844362104849039284835937688474036588340L);
	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, sc
		);
		const static size_t width = 5;
	};

	// Traditional CDF wavelets
	template<typename T>
	class CDF9_7{
		constexpr static auto s1 = detail::repeat_update_d<T, 2>(0, -1.58613434205992355842831545133740131985598525529112656778527L);
		constexpr static auto s2 = detail::repeat_update_s<T, 2>(-1, -0.0529801185729614146241295675034771089920773921055534971880099L);
		constexpr static auto s3 = detail::repeat_update_d<T, 2>(0, 0.882911075530933295919790099002837930341944716149740640164429L);
		constexpr static auto s4 = detail::repeat_update_s<T, 2>(-1, 0.443506852043971152115604215168913719478036852964167569567164L);
		constexpr static auto sc = detail::ScaleStep<T>(1.14960439886024115979507564219148965843436742907448991688182L);
	public:
		using type = T;

		constexpr static auto steps = std::make_tuple(
			s1, s2, s3, s4, sc
		);
		const static size_t width = 9;
	};

namespace detail{
	template<Wavelet WVLT, typename T> struct wavelet_from_enum;

	#define X(name, value) \
	template<typename T> struct wavelet_from_enum<Wavelet::name, T> { \
		using type = name<T>; \
	};
	LIFTED_WAVELETS
	#undef X

	template<Wavelet WVLT, typename T>
	using wavelet_from_enum_t = typename wavelet_from_enum<WVLT, T>::type;
}
}

#endif