import { Brain as LucideBrain, LucideProps } from 'lucide-react';

const AiIcon = ({ className, ...props }: LucideProps) => {
  return <LucideBrain className={className} {...props} />;
};

export default AiIcon;